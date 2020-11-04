import logging
from functools import partial
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult, SliceAnomalyDetectionResult
from uncertify.evaluation.thresholding import threshold_vs_fpr
from uncertify.evaluation.thresholding import calculate_fpr_minus_accepted
from uncertify.evaluation.model_performance import mean_std_dice_scores
from uncertify.evaluation.model_performance import mean_std_iou_scores
from uncertify.algorithms.golden_section_search import golden_section_search
from uncertify.evaluation.inference import yield_anomaly_predictions
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.evaluation.inference import SliceWiseCriteria
from uncertify.visualization.threshold_search import plot_fpr_vs_residual_threshold
from uncertify.visualization.model_performance import plot_segmentation_performance_vs_threshold
from uncertify.visualization.model_performance import plot_confusion_matrix
from uncertify.visualization.model_performance import plot_roc_curve, plot_precision_recall_curve
from uncertify.visualization.histograms import plot_loss_histograms
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Iterable

LOG = logging.getLogger(__name__)

OUT_DIR_PATH = DATA_DIR_PATH / 'evaluation'


def run_evaluation_pipeline(model: nn.Module,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader,
                            eval_cfg: EvaluationConfig,
                            best_threshold: float = None,
                            run_segmentation: bool = True,
                            run_anomaly_detection: bool = True,
                            run_histograms: bool = True) -> EvaluationResult:
    """Main function which runs the complete evaluation pipeline for a trained model and a test dataset."""
    results = EvaluationResult(out_dir_path=OUT_DIR_PATH)
    results.make_dirs()

    with torch.no_grad():
        if best_threshold is None:  # To skip lengthy calculation if we want to set it by hand
            results = run_residual_threshold_evaluation(model, train_dataloader, eval_cfg, results)
        else:
            results.pixel_anomaly_result.best_threshold = best_threshold

        if run_segmentation:
            results = run_segmentation_performance(eval_cfg, results, val_dataloader, model)

        if run_anomaly_detection:
            results = run_pixel_wise_anomaly_detection_performance(eval_cfg, model, val_dataloader, results)

        if run_histograms:
            results = run_loss_term_histograms(model, train_dataloader, val_dataloader, eval_cfg, results)

        # results.to_json()
        return results


def run_residual_threshold_evaluation(model: nn.Module, train_dataloader: DataLoader,
                                      eval_cfg: EvaluationConfig,
                                      results: EvaluationResult) -> EvaluationResult:
    """Search for best threshold given an accepted FPR and update the results dict."""
    thresh_cfg = eval_cfg.thresh_search_config
    pixel_thresholds = np.linspace(thresh_cfg.min_val, thresh_cfg.max_val, thresh_cfg.num_values)
    LOG.info(f'Determining best residual threshold based on accepted FPR ({thresh_cfg.accepted_fpr}) '
             f'checking on pixel thresholds {list(pixel_thresholds)}...')
    thresholds, train_false_positive_rates = threshold_vs_fpr(train_dataloader, model,
                                                              thresholds=pixel_thresholds,
                                                              use_ground_truth=False,
                                                              n_batches_per_thresh=thresh_cfg.num_batches)
    objective = partial(calculate_fpr_minus_accepted,
                        accepted_fpr=thresh_cfg.accepted_fpr,
                        data_loader=train_dataloader,
                        model=model,
                        use_ground_truth=False,
                        n_batches_per_thresh=thresh_cfg.num_batches)
    best_threshold = golden_section_search(objective,
                                           low=thresh_cfg.gss_lower_val,
                                           up=thresh_cfg.gss_upper_val,
                                           tolerance=thresh_cfg.gss_tolerance,
                                           return_mean=True)
    results.best_threshold = best_threshold
    fpr_vs_threshold_fig = plot_fpr_vs_residual_threshold(accepted_fpr=thresh_cfg.accepted_fpr,
                                                          calculated_threshold=best_threshold,
                                                          thresholds=pixel_thresholds,
                                                          fpr_train=train_false_positive_rates)
    fpr_vs_threshold_fig.savefig(results.plot_dir_path / 'fpr_vs_threshold.png')
    LOG.info(f'Calculated residual threshold: {best_threshold}')
    return results


def run_segmentation_performance(eval_cfg: EvaluationConfig, results: EvaluationResult,
                                 val_dataloader: DataLoader, model: nn.Module) -> EvaluationResult:
    perf_cfg = eval_cfg.performance_config

    # Calculate scores for all pixel thresholds
    pixel_thresholds = np.linspace(perf_cfg.min_val, perf_cfg.max_val, perf_cfg.num_values)
    LOG.info(f'Calculating segmentation (DICE / IOU) performance for residual thresholds: {list(pixel_thresholds)}')
    mean_dice_scores, std_dice_score = mean_std_dice_scores(val_dataloader, model,
                                                            pixel_thresholds, eval_cfg.use_n_batches)
    if perf_cfg.do_iou:
        mean_iou_scores, std_iou_scores = mean_std_iou_scores(val_dataloader, model,
                                                              pixel_thresholds, eval_cfg.use_n_batches)
    else:
        mean_iou_scores, std_iou_scores = None, None
    # If best threshold is calculated already, calculate segmentation score as well
    if results.pixel_anomaly_result.best_threshold is not None:
        best_mean_dice_score, best_std_dice_score = mean_std_dice_scores(val_dataloader, model,
                                                                         [results.pixel_anomaly_result.best_threshold],
                                                                         eval_cfg.use_n_batches)
        results.per_patient_dice_score_mean = best_mean_dice_score[0]
        results.per_patient_dice_score_std = best_std_dice_score[0]

    segmentation_performance_fig = plot_segmentation_performance_vs_threshold(thresholds=pixel_thresholds,
                                                                              dice_scores=mean_dice_scores,
                                                                              dice_stds=std_dice_score,
                                                                              iou_scores=mean_iou_scores,
                                                                              iou_stds=std_iou_scores,
                                                                              train_set_threshold=results.pixel_anomaly_result.best_threshold)
    segmentation_performance_fig.savefig(results.plot_dir_path / 'seg_performance_vs_thresh.png')
    return results


def run_pixel_wise_anomaly_detection_performance(eval_config: EvaluationConfig, model: nn.Module,
                                                 val_dataloader: DataLoader,
                                                 results: EvaluationResult) -> EvaluationResult:
    """Perform pixel-level anomaly detection performance evaluation."""
    LOG.info('Calculating pixel-wise anomaly detection performance (ROC, PRC)...')

    anomaly_scores = yield_anomaly_predictions(val_dataloader, model,
                                               eval_config.use_n_batches, results.pixel_anomaly_result.best_threshold)

    # Step 1 - Pixel-wise anomaly detection
    y_true = anomaly_scores.pixel_wise.y_true
    y_pred_proba = anomaly_scores.pixel_wise.y_pred_proba
    fpr, tpr, threshs, au_roc = calculate_roc(y_true, y_pred_proba)
    precision, recall, threshs, au_prc = calculate_prc(y_true, y_pred_proba)
    results.pixel_anomaly_result.au_prc = au_prc
    results.pixel_anomaly_result.au_roc = au_roc

    roc_fig = plot_roc_curve(fpr, tpr, au_roc,
                             # calculated_threshold=results.pixel_anomaly_result.best_threshold,
                             # thresholds=threshs,
                             title=f'ROC Curve Pixel-wise Anomaly Detection', figsize=(6, 6))
    roc_fig.savefig(results.plot_dir_path / f'pixel_wise_roc.png')

    prc_fig = plot_precision_recall_curve(recall, precision, au_prc,
                                          # calculated_threshold=results.pixel_anomaly_result.best_threshold,
                                          # thresholds=threshs,
                                          title=f'PR Curve Pixel-wise Anomaly Detection', figsize=(6, 6))
    prc_fig.savefig(results.plot_dir_path / 'pixel_wise_prc.png')

    if anomaly_scores.pixel_wise.y_pred is not None:
        conf_matrix = confusion_matrix(anomaly_scores.pixel_wise.y_true, anomaly_scores.pixel_wise.y_pred)
        confusion_matrix_fig, _ = plot_confusion_matrix(conf_matrix, categories=['normal', 'anomaly'],
                                                        cbar=False, cmap='YlOrRd_r', figsize=(6, 6))
        confusion_matrix_fig.savefig(results.plot_dir_path / 'pixel_wise_confusion_matrix.png')
    # TODO: Calculate Recall, Precision, Accuracy, F1 score from confusion matrix.

    # Step 2 - Slice-wise anomaly detection
    LOG.info('Calculating slice-wise anomaly detection performance (ROC, PRC)...')

    for criteria in SliceWiseCriteria:
        y_true = anomaly_scores.slice_wise[criteria.name].anomaly_score.y_true
        print(len(y_true))
        y_pred_proba = anomaly_scores.slice_wise[criteria.name].anomaly_score.y_pred_proba

        fpr, tpr, _, au_roc = calculate_roc(y_true, y_pred_proba)
        precision, recall, _, au_prc = calculate_prc(y_true, y_pred_proba)
        anomaly_result_for_criteria = SliceAnomalyDetectionResult()
        anomaly_result_for_criteria.criteria = str(criteria)
        anomaly_result_for_criteria.au_prc = au_prc
        anomaly_result_for_criteria.au_roc = au_roc
        results.slice_anomaly_results.results.append(anomaly_result_for_criteria)

        roc_fig = plot_roc_curve(fpr, tpr, au_roc,
                                 title=f'ROC Curve Sample-wise Anomaly Detection [{str(criteria)}]',
                                 figsize=(6, 6))
        roc_fig.savefig(results.plot_dir_path / f'sample_wise_roc_{str(criteria)}.png')

        prc_fig = plot_precision_recall_curve(recall, precision, au_prc,
                                              title=f'PR Curve Sample-wise Anomaly Detection [{str(criteria)}]',
                                              figsize=(6, 6))
        prc_fig.savefig(results.plot_dir_path / f'sample_wise_prc_{str(criteria)}.png')

    return results


def calculate_roc(y_true: Iterable, y_pred_proba: Iterable) -> Tuple[Iterable, Iterable, Iterable, float]:
    """Calculate the roc curve for true labels and model predictions.
    Returns:
        (false_positive_rates, true_positive_rates, thresholds, area_under_roc_curve)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    au_roc = roc_auc_score(y_true, y_pred_proba)
    return fpr, tpr, thresholds, au_roc


def calculate_prc(y_true: Iterable, y_pred_proba: Iterable) -> Tuple[Iterable, Iterable, Iterable, float]:
    """Calculate the prc curve for true labels and model predictions.
    Returns:
        (precisions, recalls, thresholds, area_under_prc_curve)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    au_prc = average_precision_score(y_true, y_pred_proba)
    return precision, recall, thresholds, au_prc


def run_loss_term_histograms(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                             eval_cfg: EvaluationConfig, results: EvaluationResult) -> EvaluationResult:
    LOG.info('Producing model statistics histograms...')
    val_generator = yield_inference_batches(val_dataloader, model,
                                            residual_threshold=results.pixel_anomaly_result.best_threshold,
                                            max_batches=eval_cfg.use_n_batches,
                                            progress_bar_suffix='validation set')
    train_generator = yield_inference_batches(train_dataloader, model,
                                              residual_threshold=results.pixel_anomaly_result.best_threshold,
                                              max_batches=eval_cfg.use_n_batches,
                                              progress_bar_suffix='training set')

    figs_axes = plot_loss_histograms(output_generators=[train_generator, val_generator],
                                     names=['Training Set', 'Validation Set'],  # TODO: Use dataset names
                                     figsize=(12, 6), ylabel='Normalized Frequency', plot_density=True,
                                     show_data_ticks=False, kde_bandwidth=[0.009, 0.009 * 5], show_histograms=False)

    for idx, (fig, _) in enumerate(figs_axes):
        fig.savefig(results.plot_dir_path / f'loss_term_distributions_{idx}.png')

    return results

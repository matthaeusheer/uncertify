import logging
from functools import partial
import random
from pathlib import Path
import json

import yaml
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
from uncertify.evaluation.ood_metrics import sample_wise_waic_scores
from uncertify.evaluation.dose import full_pipeline_slice_wise_dose_scores, STATISTICS_FUNCTIONS
from uncertify.evaluation.inference import yield_anomaly_predictions
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.evaluation.inference import SliceWiseCriteria
from uncertify.visualization.threshold_search import plot_fpr_vs_residual_threshold
from uncertify.visualization.model_performance import plot_segmentation_performance_vs_threshold
from uncertify.visualization.model_performance import plot_confusion_matrix
from uncertify.visualization.model_performance import plot_roc_curve, plot_precision_recall_curve
from uncertify.visualization.model_performance import plot_multi_roc_curves, plot_multi_prc_curves
from uncertify.visualization.histograms import plot_loss_histograms
from uncertify.evaluation.configs import PixelAnomalyDetectionResult, SliceAnomalyDetectionResults
from uncertify.evaluation.configs import OODDetectionResult, OODDetectionResults
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Iterable, List

LOG = logging.getLogger(__name__)

OUT_DIR_PATH = DATA_DIR_PATH / 'evaluation'


def run_evaluation_pipeline(model: nn.Module,
                            train_dataloader: DataLoader,
                            val_dataloader: DataLoader,
                            eval_cfg: EvaluationConfig,
                            best_threshold: float = None,
                            run_segmentation: bool = True,
                            run_anomaly_detection: bool = True,
                            run_histograms: bool = True,
                            run_ood_detection: bool = True,
                            ensemble_models: Iterable[nn.Module] = None) -> EvaluationResult:
    """Main function which runs the complete evaluation pipeline for a trained model and a test dataset."""
    results = EvaluationResult(OUT_DIR_PATH, eval_cfg, PixelAnomalyDetectionResult(),
                               SliceAnomalyDetectionResults(), OODDetectionResults())
    results.make_dirs()
    results.test_set_name = val_dataloader.dataset.name
    LOG.info(f'Evaluation run output: {results.out_dir_path / results.run_dir_name}')

    with torch.no_grad():
        if best_threshold is None:  # To skip lengthy calculation if we want to set it by hand
            results = run_residual_threshold_evaluation(model, train_dataloader, eval_cfg, results)
        else:
            results.pixel_anomaly_result.best_threshold = best_threshold

        if run_segmentation:
            results = run_segmentation_performance(eval_cfg, results, val_dataloader, model)

        if run_anomaly_detection:
            results = run_anomaly_detection_performance(eval_cfg, model, val_dataloader, results)

        if run_histograms:
            results = run_loss_term_histograms(model, train_dataloader, val_dataloader, eval_cfg, results)

        if run_ood_detection:
            assert ensemble_models is not None, f'Need to provide list of ensemble models for WAIC OOD performance!'
            results = run_ood_detection_performance(ensemble_models, train_dataloader, val_dataloader, eval_cfg,
                                                    results)

        results.to_json()
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
                                                              n_batches_per_thresh=eval_cfg.use_n_batches)
    objective = partial(calculate_fpr_minus_accepted,
                        accepted_fpr=thresh_cfg.accepted_fpr,
                        data_loader=train_dataloader,
                        model=model,
                        use_ground_truth=False,
                        n_batches_per_thresh=eval_cfg.use_n_batches)
    best_threshold = golden_section_search(objective,
                                           low=thresh_cfg.gss_lower_val,
                                           up=thresh_cfg.gss_upper_val,
                                           tolerance=thresh_cfg.gss_tolerance,
                                           return_mean=True)
    results.best_threshold = best_threshold
    if eval_cfg.do_plots:
        fpr_vs_threshold_fig = plot_fpr_vs_residual_threshold(accepted_fpr=thresh_cfg.accepted_fpr,
                                                              calculated_threshold=best_threshold,
                                                              thresholds=pixel_thresholds,
                                                              fpr_train=train_false_positive_rates)
        fpr_vs_threshold_fig.savefig(results.plot_dir_path / 'fpr_vs_threshold.png')
    LOG.info(f'Calculated residual threshold: {best_threshold}')
    return results


def run_segmentation_performance(eval_cfg: EvaluationConfig, results: EvaluationResult,
                                 val_dataloader: DataLoader, model: nn.Module) -> EvaluationResult:
    perf_cfg = eval_cfg.seg_performance_config
    LOG.info(f'Running segmentation performance '
             f'(residual threshold = {results.pixel_anomaly_result.best_threshold})...')

    # Calculate scores for all pixel thresholds
    if perf_cfg.do_multiple_thresholds:
        pixel_thresholds = np.linspace(perf_cfg.min_val, perf_cfg.max_val, perf_cfg.num_values)
        LOG.info(f'Calculating segmentation (DICE / IOU) performance for residual thresholds: {list(pixel_thresholds)}')
        mean_dice_scores, std_dice_score = mean_std_dice_scores(val_dataloader, model,
                                                                pixel_thresholds, eval_cfg.use_n_batches)
        if perf_cfg.do_iou:
            mean_iou_scores, std_iou_scores = mean_std_iou_scores(val_dataloader, model,
                                                                  pixel_thresholds, eval_cfg.use_n_batches)
        else:
            mean_iou_scores, std_iou_scores = None, None
        if eval_cfg.do_plots:
            threshold = results.pixel_anomaly_result.best_threshold
            segmentation_performance_fig = plot_segmentation_performance_vs_threshold(thresholds=pixel_thresholds,
                                                                                      dice_scores=mean_dice_scores,
                                                                                      dice_stds=std_dice_score,
                                                                                      iou_scores=mean_iou_scores,
                                                                                      iou_stds=std_iou_scores,
                                                                                      train_set_threshold=threshold)
            segmentation_performance_fig.savefig(results.plot_dir_path / 'seg_performance_vs_thresh.png')

    # If best threshold is calculated already, calculate segmentation score as well
    if results.pixel_anomaly_result.best_threshold is not None:
        best_mean_dice_score, best_std_dice_score = mean_std_dice_scores(val_dataloader, model,
                                                                         [results.pixel_anomaly_result.best_threshold],
                                                                         eval_cfg.use_n_batches)
        results.pixel_anomaly_result.per_patient_dice_score_mean = best_mean_dice_score[0]
        results.pixel_anomaly_result.per_patient_dice_score_std = best_std_dice_score[0]

    return results


def run_anomaly_detection_performance(eval_config: EvaluationConfig, model: nn.Module,
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

    if eval_config.do_plots:
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
        y_pred_proba = anomaly_scores.slice_wise[criteria.name].anomaly_score.y_pred_proba

        fpr, tpr, _, au_roc = calculate_roc(y_true, y_pred_proba)
        precision, recall, _, au_prc = calculate_prc(y_true, y_pred_proba)
        anomaly_result_for_criteria = SliceAnomalyDetectionResult()
        anomaly_result_for_criteria.criteria = criteria.name
        anomaly_result_for_criteria.au_prc = au_prc
        anomaly_result_for_criteria.au_roc = au_roc
        results.slice_anomaly_results.results.append(anomaly_result_for_criteria)

        if eval_config.do_plots:
            roc_fig = plot_roc_curve(fpr, tpr, au_roc,
                                     title=f'ROC Curve Sample-wise Anomaly Detection [{str(criteria.name)}]',
                                     figsize=(6, 6))
            roc_fig.savefig(results.plot_dir_path / f'sample_wise_roc_{str(criteria.name)}.png')

            prc_fig = plot_precision_recall_curve(recall, precision, au_prc,
                                                  title=f'PR Curve Sample-wise Anomaly Detection '
                                                        f'[{str(criteria.name)}]',
                                                  figsize=(6, 6))
            prc_fig.savefig(results.plot_dir_path / f'sample_wise_prc_{str(criteria.name)}.png')

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

    if eval_cfg.do_plots:
        figs_axes = plot_loss_histograms(output_generators=[train_generator, val_generator],
                                         names=['Training Set', 'Validation Set'],  # TODO: Use dataset names
                                         figsize=(12, 6), ylabel='Normalized Frequency', plot_density=True,
                                         show_data_ticks=False, kde_bandwidth=[0.009, 0.009 * 5], show_histograms=False)

        for idx, (fig, _) in enumerate(figs_axes):
            fig.savefig(results.plot_dir_path / f'loss_term_distributions_{idx}.png')

    return results


def run_ood_detection_performance(model_ensemble: List[nn.Module], train_dataloader: DataLoader,
                                  val_dataloader: DataLoader, eval_cfg: EvaluationConfig,
                                  results: EvaluationResult) -> EvaluationResult:
    LOG.info('Running OOD detection performance...')
    results.test_set_name = val_dataloader.dataset.name
    min_length_dataloader = min(len(train_dataloader), len(val_dataloader))
    use_n_batches = min(eval_cfg.use_n_batches, min_length_dataloader)

    ood_metrics = {
        'waic':
            {
                'in_distribution': partial(sample_wise_waic_scores, model_ensemble, train_dataloader,
                                           results.pixel_anomaly_result.best_threshold, use_n_batches),
                'ood': partial(sample_wise_waic_scores, model_ensemble, val_dataloader,
                               results.pixel_anomaly_result.best_threshold, use_n_batches),
                'out_filter': lambda out_tuple: out_tuple
            },
        'dose':
            {
                'in_distribution': partial(full_pipeline_slice_wise_dose_scores,
                                           train_dataloader,
                                           train_dataloader,
                                           model_ensemble[0],
                                           list(STATISTICS_FUNCTIONS.keys()),
                                           use_n_batches),
                'ood': partial(full_pipeline_slice_wise_dose_scores,
                               train_dataloader,
                               val_dataloader,
                               model_ensemble[0],
                               list(STATISTICS_FUNCTIONS.keys()),
                               use_n_batches),
                'out_filter': lambda out_tuple: (out_tuple[0], out_tuple[1]['is_lesional'])
            }
    }

    for metric_name, metric_func_dict in ood_metrics.items():
        y_pred_proba_all = []  # These are the e.g. OOD scores for both in- and OOD data!
        y_true_all = []
        y_pred_proba_lesional = []
        y_true_lesional = []
        y_pred_proba_healthy = []
        y_true_healthy = []

        # First use val_dataloader
        LOG.info(f'Inference on out-of-distribution data...')
        ood_scores, lesional_list = ood_metrics[metric_name]['out_filter'](metric_func_dict['ood']())
        y_pred_proba_all.extend(ood_scores)
        y_true_all.extend(len(ood_scores) * [1.0])

        lesional_indices = [idx for idx, is_lesional in enumerate(lesional_list) if is_lesional]
        normal_indices = [idx for idx, is_lesional in enumerate(lesional_list) if not is_lesional]

        lesional_ood_scores = [ood_scores[idx] for idx in lesional_indices]
        healthy_odd_scores = [ood_scores[idx] for idx in normal_indices]

        n_lesional_samples = len(lesional_ood_scores)
        y_pred_proba_lesional.extend(lesional_ood_scores)
        y_true_lesional.extend(n_lesional_samples * [1.0])

        n_healthy_samples = len(healthy_odd_scores)
        y_pred_proba_healthy.extend(healthy_odd_scores)
        y_true_healthy.extend(n_healthy_samples * [1.0])

        # Now use train_dataloader
        LOG.info(f'Inference on in-distribution data...')
        ood_scores, _ = ood_metrics[metric_name]['out_filter'](metric_func_dict['in_distribution']())

        y_pred_proba_all.extend(ood_scores)
        y_true_all.extend(len(ood_scores) * [0.0])

        # Train dataloader has no lesional samples, so fill up lesional and healthy ones from above with same amount
        y_pred_proba_lesional.extend(random.sample(ood_scores, n_lesional_samples))
        y_true_lesional.extend(n_lesional_samples * [0.0])

        y_pred_proba_healthy.extend(random.sample(ood_scores, n_healthy_samples))
        y_true_healthy.extend(n_healthy_samples * [0.0])

        def calculate_ood_performance(mode: str, y_true: list, y_pred_proba: list) -> dict:
            if mode not in ['all', 'lesional', 'healthy']:
                raise ValueError(f'Given mode ({mode}) is not valid! Choose from all, lesional, healthy.')
            fpr, tpr, _, au_roc = calculate_roc(y_true, y_pred_proba)
            precision, recall, _, au_prc = calculate_prc(y_true, y_pred_proba)
            out_dict = {'tpr': tpr, 'fpr': fpr, 'au_roc': au_roc,
                        'precision': precision, 'recall': recall, 'au_prc': au_prc}
            return out_dict

        def update_results(ood_scores: dict, mode_: str) -> None:
            ood_result = OODDetectionResult()
            ood_result.au_roc = ood_scores['au_roc']
            ood_result.au_prc = ood_scores['au_prc']
            ood_result.mode = mode_
            results.ood_detection_results.results.append(ood_result)

        # Will be filled for each mode ('all', 'healthy', 'lesional') with
        metrics_dict = {}
        for mode, y_true, y_pred_proba in zip(['all', 'healthy', 'lesional'],
                                              [y_true_all, y_true_healthy, y_true_lesional],
                                              [y_pred_proba_all, y_pred_proba_healthy, y_pred_proba_lesional]):
            if mode in ['healthy', 'lesional'] and not sum(lesional_list) > 0:
                continue
            scores = calculate_ood_performance(mode, y_true, y_pred_proba)
            update_results(scores, mode)
            metrics_dict[mode] = scores

        if eval_cfg.do_plots:
            labels = list(metrics_dict.keys())
            fprs = [score_dict['fpr'] for score_dict in metrics_dict.values()]
            tprs = [score_dict['tpr'] for score_dict in metrics_dict.values()]
            au_rocs = [score_dict['au_roc'] for score_dict in metrics_dict.values()]
            recalls = [score_dict['recall'] for score_dict in metrics_dict.values()]
            precisions = [score_dict['precision'] for score_dict in metrics_dict.values()]
            au_prcs = [score_dict['au_prc'] for score_dict in metrics_dict.values()]
            roc_fig = plot_multi_roc_curves(fprs, tprs, au_rocs, labels,
                                            title=f'{metric_name} ROC Curve OOD Detection', figsize=(6, 6))
            prc_fig = plot_multi_prc_curves(recalls, precisions, au_prcs, labels,
                                            title=f'{metric_name} PR Curve OOD Detection', figsize=(6, 6))
            roc_fig.savefig(results.plot_dir_path / f'ood_roc_{metric_name}.png')
            prc_fig.savefig(results.plot_dir_path / f'ood_prc_{metric_name}.png')
    return results


def print_results(results: EvaluationResult) -> None:
    # Pixel-wise Anomaly Detection
    print('Pixel-wise Anomaly Detection')
    print(results.pixel_anomaly_result)
    # Slice-wise Anomaly Detection
    print('Slice-wise Anomaly Detection')
    print(results.slice_anomaly_results)
    # OOD detection
    print('OOD Detection')
    print(results.ood_detection_results)


def print_results_from_evaluation_dirs(work_dir_path: Path, run_numbers: list,
                                       print_results_only: bool = False) -> None:
    """Print the aggregated results from multiple evaluation runs."""

    def float_representer(dumper, value):
        text = '{0:.4f}'.format(value)
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', text)

    yaml.add_representer(float, float_representer)

    for run_number in run_numbers:
        eval_dir_path = work_dir_path / f'evaluation_{run_number}'
        eval_file_name = f'evaluation_results_{run_number}.json'
        print(f'--- Evaluation summary run {run_number} ---')
        with open(eval_dir_path / eval_file_name, 'r') as infile:
            results = json.load(infile)
            if print_results_only:
                results = {key: val for key, val in results.items() if 'result' in key}
            print(yaml.dump(results))

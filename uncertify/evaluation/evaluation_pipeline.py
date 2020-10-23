import logging
from functools import partial

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from scikitplot.metrics import plot_precision_recall, plot_roc

from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult, PixelThresholdSearchConfig
from uncertify.evaluation.thresholding import threshold_vs_fpr
from uncertify.evaluation.thresholding import calculate_fpr_minus_accepted
from uncertify.evaluation.model_performance import mean_std_dice_scores
from uncertify.evaluation.model_performance import mean_std_iou_scores
from uncertify.evaluation.model_performance import calculate_confusion_matrix
from uncertify.algorithms.golden_section_search import golden_section_search
from uncertify.deploy import yield_y_true_y_pred
from uncertify.deploy import yield_reconstructed_batches
from uncertify.visualization.threshold_search import plot_fpr_vs_residual_threshold
from uncertify.visualization.model_performance import plot_segmentation_performance_vs_threshold
from uncertify.visualization.model_performance import plot_confusion_matrix
from uncertify.visualization.histograms import plot_loss_histograms
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Iterable

LOG = logging.getLogger(__name__)

OUT_DIR_PATH = DATA_DIR_PATH / 'evaluation'


def run_evaluation_pipeline(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                            eval_cfg: EvaluationConfig) -> EvaluationResult:
    """Main function which runs the complete evaluation pipeline for a trained model and a test dataset."""
    results = EvaluationResult(out_dir_path=OUT_DIR_PATH)
    results.make_dirs()

    results, pixel_thresholds = run_residual_threshold_evaluation(model, train_dataloader, eval_cfg, results)

    results = run_pixel_wise_performance(model, val_dataloader, pixel_thresholds, eval_cfg, results)

    results = run_loss_term_histograms(model, train_dataloader, val_dataloader, eval_cfg, results)

    results.to_json()
    return results


def run_residual_threshold_evaluation(model: nn.Module, train_dataloader: DataLoader,
                                      eval_cfg: EvaluationConfig,
                                      results: EvaluationResult) -> Tuple[EvaluationResult, Iterable]:
    """Search for best threshold given an accepted FPR and update the results dict."""
    thresh_cfg = eval_cfg.thresh_search_config
    LOG.info(f'Determining best residual threshold based on accepted FPR ({thresh_cfg.accepted_fpr})')
    pixel_thresholds = np.linspace(thresh_cfg.min_val, thresh_cfg.max_val, thresh_cfg.num_values)
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
    return results, pixel_thresholds


def run_pixel_wise_performance(model: nn.Module, val_dataloader: DataLoader, pixel_thresholds: Iterable,
                               eval_cfg: EvaluationConfig, results: EvaluationResult) -> EvaluationResult:
    LOG.info(f'Calculating segmentation performance...')
    perf_cfg = eval_cfg.performance_config
    dice_scores = mean_std_dice_scores(val_dataloader, model, pixel_thresholds, perf_cfg.use_n_batches)
    iou_scores = mean_std_iou_scores(val_dataloader, model, pixel_thresholds, perf_cfg.use_n_batches)

    segmentation_performance_fig = plot_segmentation_performance_vs_threshold(thresholds=pixel_thresholds,
                                                                              dice_scores=dice_scores,
                                                                              iou_scores=iou_scores,
                                                                              train_set_threshold=results.best_threshold)
    segmentation_performance_fig.savefig(results.plot_dir_path / 'seg_performance_vs_thresh.png')

    # 2.2 - Pixel-wise anomaly detection / classification performance
    LOG.info('Calculating pixel-wise anomaly detection performance...')
    y_true, y_pred = yield_y_true_y_pred(val_dataloader, model, perf_cfg.use_n_batches)
    area_under_prc = average_precision_score(y_true, y_pred[:, 1])
    area_under_roc = roc_auc_score(y_true, y_pred[:, 1])
    results.au_prc = area_under_prc
    results.au_roc = area_under_roc

    ax_prc = plot_precision_recall(y_true, y_pred, figsize=(12, 8), classes_to_plot=[1], plot_micro=False,
                                   title=f'PR Curve Pixel-wise Anomaly Detection')
    ax_roc = plot_roc(y_true, y_pred, figsize=(12, 8), plot_micro=False, plot_macro=False, classes_to_plot=[1],
                      title=f'ROC Curve Pixel-wise Anomaly Detection')
    # TODO: Get figure and store.
    confusion_matrix = calculate_confusion_matrix(val_dataloader, model,
                                                  residual_threshold=results.best_threshold,
                                                  max_n_batches=perf_cfg.use_n_batches,
                                                  normalize=None)
    confusion_matrix_fig, _ = plot_confusion_matrix(confusion_matrix, categories=['normal', 'anomaly'],
                                                    cbar=False, cmap='YlGn', figsize=(12, 11))
    # TODO: Calculate Recall, Precision, Accuracy, F1 score from confusion matrix.
    return results


def run_loss_term_histograms(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                             eval_cfg: EvaluationConfig, results: EvaluationResult) -> EvaluationResult:
    val_generator = yield_reconstructed_batches(val_dataloader, model,
                                                residual_threshold=results.best_threshold,
                                                max_batches=eval_cfg.performance_config.use_n_batches,
                                                progress_bar_suffix='brats_val')
    train_generator = yield_reconstructed_batches(train_dataloader, model,
                                                  residual_threshold=results.best_threshold,
                                                  max_batches=eval_cfg.performance_config.use_n_batches,
                                                  progress_bar_suffix='camcan_val')

    figs_axes = plot_loss_histograms(output_generators=[train_generator, val_generator],
                                     names=['Training Set', 'Validation Set'],  # TODO: Use dataset names
                                     figsize=(12, 6), ylabel='Normalized Frequency', plot_density=True,
                                     show_data_ticks=False, kde_bandwidth=[0.009, 0.009 * 5], show_histograms=False)

    for idx, (fig, _) in enumerate(figs_axes):
        fig.savefig(results.plot_dir_path / f'loss_term_distributions_{idx}.png')

    return results

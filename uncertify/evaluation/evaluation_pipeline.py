import logging
from functools import partial

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from scikitplot.metrics import plot_precision_recall, plot_roc

from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult
from uncertify.evaluation.thresholding import threshold_vs_fpr
from uncertify.evaluation.thresholding import calculate_fpr_minus_accepted
from uncertify.evaluation.model_performance import calculate_mean_dice_scores
from uncertify.evaluation.model_performance import calculate_mean_iou_scores
from uncertify.evaluation.model_performance import calculate_confusion_matrix
from uncertify.algorithms.golden_section_search import golden_section_search
from uncertify.deploy import yield_y_true_y_pred
from uncertify.deploy import yield_reconstructed_batches
from uncertify.visualization.threshold_search import plot_fpr_vs_residual_threshold
from uncertify.visualization.model_performance import plot_segmentation_performance_vs_threshold
from uncertify.visualization.model_performance import plot_confusion_matrix
from uncertify.visualization.histograms import plot_loss_histograms
from uncertify.common import DATA_DIR_PATH

LOG = logging.getLogger(__name__)


def run_evaluation_pipeline(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                            eval_cfg: EvaluationConfig) -> EvaluationResult:
    """Main function which runs the complete evaluation pipeline for a trained model and a test dataset.

    The processing pipeline consists of the following steps:
        - Determine best threshold for an accepted false positive rate
        -
    """
    results = EvaluationResult(evaluation_config=eval_cfg,
                               out_dir_path=DATA_DIR_PATH / 'evaluation')
    results.out_dir_path.mkdir(exist_ok=True)

    # Step 1 - Calculate the best residual threshold by lowering the difference between fpr and accepted fpr
    thresh_cfg = eval_cfg.thresh_search_config
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
    best_thresholds = golden_section_search(objective,
                                            low=thresh_cfg.gss_lower_val,
                                            up=thresh_cfg.gss_upper_val,
                                            tolerance=thresh_cfg.gss_tolerance)
    best_threshold = float(np.mean(best_thresholds))
    results.best_threshold = best_threshold
    fpr_vs_threshold_fig = plot_fpr_vs_residual_threshold(accepted_fpr=thresh_cfg.accepted_fpr,
                                                          calculated_threshold=best_threshold,
                                                          thresholds=pixel_thresholds,
                                                          fpr_train=train_false_positive_rates)
    fpr_vs_threshold_fig.savefig(results.out_dir_path / results.plot_dir_name / 'fpr_vs_threshold.png')
    LOG.info(f'Calculated residual threshold: {best_threshold}')

    # Step 2 - Calculate model performance metrics
    # 2.1 - Segmentation performance
    perf_cfg = eval_cfg.performance_config
    dice_scores = calculate_mean_dice_scores(val_dataloader, model, pixel_thresholds, perf_cfg.use_n_batches)
    iou_scores = calculate_mean_iou_scores(val_dataloader, model, pixel_thresholds, perf_cfg.use_n_batches)

    segmentation_performance_fig = plot_segmentation_performance_vs_threshold(thresholds=pixel_thresholds,
                                                                              dice_scores=dice_scores,
                                                                              iou_scores=iou_scores,
                                                                              train_set_threshold=best_threshold)
    segmentation_performance_fig.savefig(results.out_dir_path / results.plot_dir_name / 'seg_performance_vs_thresh.png')

    # 2.2 - Pixel-wise anomaly detection / classification performance
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
                                                  residual_threshold=best_threshold,
                                                  max_n_batches=perf_cfg.use_n_batches,
                                                  normalize=None)
    confusion_matrix_fig, _ = plot_confusion_matrix(confusion_matrix, categories=['normal', 'anomaly'],
                                                    cbar=False, cmap='YlGn', figsize=(12, 11))
    # TODO: Calculate Recall, Precision, Accuracy, F1 score from confusion matrix.

    # Step 3) - Likelihood and other term distributions
    val_generator = yield_reconstructed_batches(val_dataloader, model,
                                                residual_threshold=best_threshold,
                                                max_batches=perf_cfg.use_n_batches,
                                                progress_bar_suffix='brats_val')
    train_generator = yield_reconstructed_batches(train_dataloader, model,
                                                  residual_threshold=best_threshold,
                                                  max_batches=perf_cfg.use_n_batches,
                                                  progress_bar_suffix='camcan_val')

    figs_axes = plot_loss_histograms(output_generators=[train_generator, val_generator],
                                     names=['Training Set', 'Validation Set'],  # TODO: Use dataset names
                                     figsize=(12, 6), ylabel='Normalized Frequency', plot_density=True,
                                     show_data_ticks=False, kde_bandwidth=[0.009, 0.009 * 5], show_histograms=False)

    for idx, (fig, _) in enumerate(figs_axes):
        fig.savefig(results.out_dir_path / results.plot_dir_name / f'loss_term_distributions_{idx}.png')

    # Step 4) - OOD detection
    # TODO: Implement OOD detection metrics.

    results.to_json()  # Writes results to output folder
    return results


def pixel_wise_performance_evaluation() -> None:
    pass


def sample_wise_performance_evaluation() -> None:
    pass

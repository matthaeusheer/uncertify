import numpy as np
import torch
from sklearn import metrics as sklearn_metrics

from uncertify.data.dataloaders import DataLoader
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.metrics.classification import dice, intersection_over_union

from typing import List, Tuple, Iterable

VALID_SEGMENTATION_SCORE_TYPES = {'dice', 'iou'}


def mean_std_dice_score(data_loader: DataLoader, model: torch.nn.Module, residual_threshold: float,
                        max_n_batches: int = None) -> Tuple[float, float]:
    """Calculate the mean and std of dice scores for batches for one residual threshold."""
    return mean_std_seg_score(data_loader, model, residual_threshold, 'dice', max_n_batches)


def mean_std_dice_scores(data_loader: DataLoader, model: torch.nn.Module, residual_thresholds: Iterable[float],
                         max_n_batches: int = None) -> Tuple[List[float], List[float]]:
    """Calculate the mean and std of dice scores for batches for multiple residual thresholds."""
    return mean_std_segmentation_scores(data_loader, model, residual_thresholds, 'dice', max_n_batches)


def mean_std_iou_score(data_loader: DataLoader, model: torch.nn.Module, residual_threshold: float,
                       max_n_batches: int = None) -> Tuple[float, float]:
    """Calculate the mean and std of iou scores for batches for one residual threshold."""
    return mean_std_seg_score(data_loader, model, residual_threshold, 'iou', max_n_batches)


def mean_std_iou_scores(data_loader: DataLoader, model: torch.nn.Module, residual_thresholds: Iterable[float],
                        max_n_batches: int = None) -> Tuple[List[float], List[float]]:
    """Calculate the mean and std of iou scores for batches for multiple residual thresholds."""
    return mean_std_segmentation_scores(data_loader, model, residual_thresholds, 'iou', max_n_batches)


def mean_std_seg_score(data_loader: DataLoader, model: torch.nn.Module, residual_threshold: float,
                       score_type: str = 'dice', max_n_batches: int = None) -> Tuple[float, float]:
    """Calculate the mean (over multiple / all batches) segmentation score for a given residual threshold.
    Arguments:
        data_loader: a uncertify data loader which yields dicts (with 'scan', 'mask', etc.)
        model: a trained pytorch model
        residual_threshold: the threshold from 0 to 1 for pixel-wise anomaly detection
        score_type: either 'dice' or 'iou'
        max_n_batches: if not None, take first max_n_batches only from the data_loader for calculation
    Returns:
        a tuple of (mean_seg_score, std_seg_score) for one threshold across multiple batches
    """
    if score_type not in VALID_SEGMENTATION_SCORE_TYPES:
        raise ValueError(f'Provided score_type ({score_type}) invalid. Choose from: {VALID_SEGMENTATION_SCORE_TYPES}')
    batch_generator = yield_inference_batches(data_loader, model, max_n_batches, residual_threshold)
    per_batch_scores = []
    for batch_idx, batch in enumerate(batch_generator):
        prediction_batch = batch.residuals_thresholded[batch.mask]
        ground_truth_batch = batch.segmentation[batch.mask]
        with torch.no_grad():
            if score_type == 'dice':
                score = dice(prediction_batch.numpy(), ground_truth_batch.numpy())
            elif score_type == 'iou':
                score = intersection_over_union(prediction_batch.numpy(), ground_truth_batch.numpy())
            else:
                raise RuntimeError(f'Arrived at a score_type ({score_type}) which is invalid. Should not happen.')
            per_batch_scores.append(score)
    # When a batch represents a single patient this actually gives back the mean and std for patient-wise scores
    seg_score_mean = float(np.mean(per_batch_scores))
    seg_score_std = float(np.std(per_batch_scores))
    return seg_score_mean, seg_score_std


def mean_std_segmentation_scores(data_loader: DataLoader, model: torch.nn.Module,
                                 residual_thresholds: Iterable[float], score_type: str = 'dice',
                                 max_n_batches: int = None) -> Tuple[List[float], List[float]]:
    """Similar to calculate_mean_segmentation_score but this time for multiple residual thresholds."""
    scores_means = []
    scores_stds = []
    for threshold in residual_thresholds:
        mean, std = mean_std_seg_score(data_loader, model, threshold, score_type, max_n_batches)
        scores_means.append(mean)
        scores_stds.append(std)
    return scores_means, scores_stds


def calculate_confusion_matrix(data_loader: DataLoader, model: torch.nn.Module, residual_threshold: float,
                               max_n_batches: int = None, normalize: bool = False) -> np.ndarray:
    """Calculate the confusion matrix for a given threshold over multiple batches of data.

    The layout of the confusion matrix follows the convention by scikit-learn, which is used to calculate sub matrices!
    """
    batch_generator = yield_inference_batches(data_loader, model, max_n_batches, residual_threshold)
    confusion_matrix = np.zeros((2, 2))  # initialize zero confusion matrix
    for batch_idx, batch in enumerate(batch_generator):
        with torch.no_grad():
            y_pred = batch.residuals_thresholded[batch.mask].flatten().numpy()
            y_true = batch.segmentation[batch.mask].flatten().numpy()
            sub_confusion_matrix = sklearn_metrics.confusion_matrix(y_true, y_pred, normalize=normalize)
            confusion_matrix += sub_confusion_matrix
    return confusion_matrix

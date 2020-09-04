import numpy as np

from typing import Tuple


def true_positives(prediction: np.ndarray, ground_truth: np.ndarray) -> int:
    return int(np.sum(np.logical_and(prediction == 1, ground_truth == 1)))


def false_positives(prediction: np.ndarray, ground_truth: np.ndarray) -> int:
    return int(np.sum(np.logical_and(prediction == 1, ground_truth == 0)))


def true_negatives(prediction: np.ndarray, ground_truth: np.ndarray) -> int:
    return int(np.sum(np.logical_and(prediction == 0, ground_truth == 0)))


def false_negatives(prediction: np.ndarray, ground_truth: np.ndarray) -> int:
    return int(np.sum(np.logical_and(prediction == 0, ground_truth == 1)))


def intersection_over_union(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted_pixels = np.sum(prediction.flatten())
    ground_truth_pixels = np.sum(ground_truth.flatten())
    intersection = np.sum(np.multiply(prediction.flatten(), ground_truth.flatten()))
    union = np.sum(predicted_pixels + ground_truth_pixels - intersection)
    iou_score = intersection / union
    return iou_score


def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted_pixels = np.sum(prediction.flatten())
    ground_truth_pixels = np.sum(ground_truth.flatten())
    intersection = np.sum(np.multiply(prediction.flatten(), ground_truth.flatten()))
    dice_score = (2 * intersection) / (predicted_pixels + ground_truth_pixels)
    return dice_score


def confusion_matrix(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Calculate the entries of a binary confusion matrix."""
    tp = true_positives(prediction, ground_truth)
    fp = false_positives(prediction, ground_truth)
    tn = true_negatives(prediction, ground_truth)
    fn = false_negatives(prediction, ground_truth)
    return np.ndarray([[tp, fp], [tn, fn]])


def true_positive_rate(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. recall or sensitivity. From the actual positives, how many did I classify as positive?"""
    tp = true_positives(prediction, ground_truth)
    fn = false_negatives(prediction, ground_truth)
    return tp / (tp + fn)


def false_positive_rate(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. fall-out or 1-specificity. From the actual negatives, how many did I falsely classified as positive?"""
    fp = false_positives(prediction, ground_truth)
    tn = true_negatives(prediction, ground_truth)
    return fp / (fp + tn)


def precision(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """From the ones that I classified as positive, how many are actually positive?"""
    tp = true_positives(prediction, ground_truth)
    fp = false_positives(prediction, ground_truth)
    return tp / (tp + fp)


def recall(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. true positive rate or sensitivity."""
    return true_positive_rate(prediction, ground_truth)

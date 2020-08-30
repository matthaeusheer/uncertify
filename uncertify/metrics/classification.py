import numpy as np

from typing import Tuple


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


def confusion_matrix(prediction: np.ndarray, ground_truth: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate the entries of a binary confusion matrix."""
    true_positives = float(np.sum(np.multiply(prediction.flatten(), ground_truth.flatten())))
    false_positives = float(np.sum(np.multiply(prediction.flatten(), np.invert(ground_truth.flatten()))))
    false_negatives = float(np.sum(np.multiply(np.invert(prediction.flatten()), ground_truth.flatten())))
    true_negatives = float(np.sum(np.multiply(np.invert(prediction.flatten()), np.invert(ground_truth.flatten()))))
    return true_positives, false_positives, true_negatives, false_negatives


def true_positive_rate(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. recall or sensitivity. From the actual positives, how many did I classify as positive?"""
    true_positives = np.sum(np.multiply(prediction.flatten(), ground_truth.flatten()))
    false_negatives = np.sum(np.multiply(np.invert(prediction.flatten()), ground_truth.flatten()))
    return true_positives / (true_positives + false_negatives)


def false_positive_rate(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. fall-out or 1-specificity. From the actual negatives, how many did I falsely classified as positive?"""
    true_positives = np.sum(np.multiply(prediction.flatten(), ground_truth.flatten()))
    false_positives = np.sum(np.multiply(prediction.flatten(), np.invert(ground_truth.flatten())))
    return false_positives / (false_positives + true_positives)


def precision(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """From the ones that I classified as positive, how many are actually positive?"""
    true_positives = np.sum(np.multiply(prediction.flatten(), ground_truth.flatten()))
    false_positives = np.sum(np.multiply(prediction.flatten(), np.invert(ground_truth.flatten())))
    return true_positives / (true_positives + false_positives)


def recall(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """A.k.a. true positive rate or sensitivity."""
    return true_positive_rate(prediction, ground_truth)

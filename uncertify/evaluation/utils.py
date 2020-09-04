import torch

from uncertify.utils.custom_types import Tensor


def threshold_batch_to_one_zero(tensor: Tensor, threshold: float) -> Tensor:
    """Apply threshold, s.t. output values become zero if smaller then threshold and one if bigger than threshold."""
    zeros = torch.zeros_like(tensor)
    ones = torch.ones_like(tensor)
    return torch.where(tensor > threshold, ones, zeros)


def is_abnormal(segmentation: Tensor, pixel_fraction_threshold: float) -> bool:
    """Based on the fraction of abnormally labelled pixels, is this scan considered abnormal?
    Args:
        segmentation: a tensor for the ground truth segmentation, expected to have only zero and one (anomaly) entries
        pixel_fraction_threshold:
    """

import torch

from uncertify.utils.custom_types import Tensor


def residual_l1_max(reconstruction: Tensor, original: Tensor) -> Tensor:
    """Construct l1 difference between original and reconstruction.

    Note: Only positive values in the residual are considered, i.e. values below zero are clamped.
    That means only cases where bright pixels which are brighter in the input (likely lesions) are kept."""
    residual = original - reconstruction
    return torch.where(residual > 0.0, residual, torch.zeros_like(residual))


def threshold_batch_to_one_zero(tensor: Tensor, threshold: float) -> Tensor:
    """Apply threshold, s.t. output values become zero if smaller then threshold and one if bigger than threshold."""
    zeros = torch.zeros_like(tensor)
    ones = torch.ones_like(tensor)
    return torch.where(tensor > threshold, ones, zeros)


def convert_segmentation_to_one_zero(segmentation: Tensor) -> Tensor:
    """The segmentation map might have multiple labels. Here we crush them to simply 1 (anomaly) or zero (healthy)."""
    return torch.where(segmentation > 0, torch.ones_like(segmentation), torch.zeros_like(segmentation))

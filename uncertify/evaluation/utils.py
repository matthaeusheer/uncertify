import torch

from uncertify.utils.custom_types import Tensor


def threshold_batch_to_one_zero(tensor: Tensor, threshold: float) -> Tensor:
    """Apply threshold, s.t. output values become zero if smaller then threshold and one if bigger than threshold."""
    zeros = torch.zeros_like(tensor)
    ones = torch.ones_like(tensor)
    return torch.where(tensor > threshold, ones, zeros)


def golden_section_search(objective, interval, tolerance):
    ...
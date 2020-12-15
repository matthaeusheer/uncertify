import torch
import scipy.ndimage

from uncertify.utils.custom_types import Tensor


def residual_l1_max(reconstruction: Tensor, original: Tensor) -> Tensor:
    """Construct l1 difference between original and reconstruction.

    Note: Only positive values in the residual are considered, i.e. values below zero are clamped.
    That means only cases where bright pixels which are brighter in the input (likely lesions) are kept."""
    residual = original - reconstruction
    return torch.where(residual > 0.0, residual, torch.zeros_like(residual))


def residual_l1(reconstruction: Tensor, original: Tensor) -> Tensor:
    """Construct the absolute l1 difference between original and reconstruction images."""
    return torch.abs_(original - reconstruction)


def mask_background_to_zero(input_tensor: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, input_tensor, torch.zeros_like(input_tensor))


def mask_background_to_value(input_tensor: Tensor, mask: Tensor, value: float) -> Tensor:
    return torch.where(mask, input_tensor, value * torch.ones_like(input_tensor))


def threshold_batch_to_one_zero(tensor: Tensor, threshold: float) -> Tensor:
    """Apply threshold, s.t. output values become zero if smaller then threshold and one if bigger than threshold."""
    zeros = torch.zeros_like(tensor)
    ones = torch.ones_like(tensor)
    return torch.where(tensor > threshold, ones, zeros)


def convert_segmentation_to_one_zero(segmentation: Tensor) -> Tensor:
    """The segmentation map might have multiple labels. Here we crush them to simply 1 (anomaly) or zero (healthy)."""
    return torch.where(segmentation > 0, torch.ones_like(segmentation), torch.zeros_like(segmentation))


def erode_mask(mask: Tensor) -> Tensor:
    """Erode the boolean mask tensor inwards the get rid of edge effects on the residual mask."""
    dev = mask.device()
    mask = mask.cpu()
    mask = scipy.ndimage.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=12)
    mask = torch.tensor(mask.cuda())
    return mask

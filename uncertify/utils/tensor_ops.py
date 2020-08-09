import torch

from uncertify.utils.custom_types import Tensor


def normalize_to_0_1(tensor: Tensor) -> Tensor:
    """Takes a pytorch tensor and normalizes the values to the range [0, 1], i.e. largest value gets 1, smallest 0."""
    tensor = tensor - tensor.min()
    return tensor / (tensor.max() - tensor.min())


def get_mean_and_std(tensor: Tensor) -> Tensor:
    return torch.mean(tensor), torch.std(tensor)


def get_min_and_max(tensor: Tensor) -> Tensor:
    return torch.min(tensor), torch.max(tensor)

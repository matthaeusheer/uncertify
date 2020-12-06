import math

import scipy.stats
import torch

from uncertify.utils.custom_types import Tensor

from typing import Union


def normalize_to_0_1(tensor: Tensor, min_val: Union[float, torch.tensor] = None,
                     max_val: Union[float, torch.tensor] = None) -> Tensor:
    """Takes a pytorch tensor and normalizes the values to the range [0, 1], i.e. largest value gets 1, smallest 0."""
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def get_mean_and_std(tensor: Tensor) -> Tensor:
    return torch.mean(tensor), torch.std(tensor)


def get_min_and_max(tensor: Tensor) -> Tensor:
    return torch.min(tensor), torch.max(tensor)


def print_scipy_stats_description(description_result: scipy.stats.stats.DescribeResult,
                                  name: str = None, column_spacer: int = 25) -> None:
    name_part = f'Name: {name}' if name is not None else ''
    min_part = f'min: {description_result.minmax[0]:.2f}'
    max_part = f'max: {description_result.minmax[1]:.2f}'
    mean_part = f'mean: {description_result.mean:.2f}'
    std_part = f'std: {math.sqrt(description_result.variance):.2f}'

    print(f''.join([f'{part:{column_spacer}}' for part in [name_part, min_part, max_part, mean_part, std_part]]))

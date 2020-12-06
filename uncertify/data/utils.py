import math
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_n_masked_pixels(data_loader: DataLoader, max_n_batches: int = None) -> int:
    """Get the total number of masked pixels in the complete dataloader dataset."""
    n_masked_pixels = 0
    for batch in itertools.islice(data_loader, max_n_batches) if max_n_batches is not None else data_loader:
        n_masked_pixels += int(torch.sum(batch['mask']))
    return n_masked_pixels


def gauss_2d_tensor_image(grid_size, std, y_offset=0, x_offset=0, normalize: bool = False) -> torch.tensor:
    """Create a 2D grid of size grid_size x grid_size with a gaussian blob in it. Offsets w.r.t. image center."""
    kernel_size = grid_size
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean_y = ((kernel_size - 1) + y_offset) / 2.
    mean_x = ((kernel_size - 1) + x_offset) / 2.
    variance = std ** 2.
    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - torch.tensor([mean_x, mean_y])) ** 2., dim=-1) / (2 * variance))
    # Make sure sum of values in gaussian kernel equals 1.
    if normalize:
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    return gaussian_kernel.view(kernel_size, kernel_size)


def gaussian(x, mu, sig):
    """1D Gaussian"""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

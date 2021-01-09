"""
Operations to be performed on pytorch tensors representing images.
"""

import random

import numpy as np
import torch

from uncertify.data.utils import gauss_2d_tensor_image


def add_circle(tensor: torch.Tensor, center: dict, radius: float):
    """Adds a circle at center and radius on a tensor image."""
    height, width = tensor.shape
    xx, yy = np.mgrid[:width, :height]
    circle = (xx - center['x']) ** 2 + (yy - center['y']) ** 2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    c = torch.FloatTensor(circle < radius**2)
    tensor += c
    return tensor


def add_random_circles(tensor: torch.Tensor, n_circles: int, equalize_overlaps: bool = True):
    """Adds n_circles random circles onto the image."""
    height, width = tensor.shape
    circle_img = torch.zeros_like(tensor)
    for _ in range(n_circles):
        circle_img = add_circle(circle_img, {'x': random.randint(0, width), 'y': random.randint(0, height)}, random.randint(1, int(max(height, width) / 20)))
    tensor += (circle_img != 0)
    if equalize_overlaps:
        tensor = (tensor != 0)
    return tensor.type(torch.FloatTensor)


def add_random_gauss_blob(tensor: torch.Tensor) -> torch.Tensor:
    """Adds a gaussian blob to the image with a mean and standard deviation."""
    height, _ = tensor.shape
    std = random.randint(3, 30)
    x_offset = random.randint(0, height // 2)
    y_offset = random.randint(0, height // 2)
    gauss_image = gauss_2d_tensor_image(height, std, y_offset, x_offset, normalize=False)
    tensor += gauss_image
    return tensor


def add_random_gauss_blobs(tensor: torch.Tensor, n_blobs: int) -> torch.Tensor:
    for _ in range(n_blobs):
        tensor = add_random_gauss_blob(tensor)
    return tensor

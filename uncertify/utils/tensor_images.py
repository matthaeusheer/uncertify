"""
Operations to be performed on pytorch tensors representing images.
"""

import random

import numpy as np
import torch

from uncertify.data.utils import gauss_2d_tensor_image, gaussian

from typing import Tuple, Optional


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
        circle_img = add_circle(circle_img, {'x': random.randint(0, width), 'y': random.randint(0, height)}, random.randint(1, int(max(height, width) / 30)))
    tensor += (circle_img != 0)
    if equalize_overlaps:
        tensor = (tensor != 0)
    return tensor.type(torch.FloatTensor)


def add_random_gauss_blob(tensor: torch.Tensor, weight: float = 1000) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adds a gaussian blob to the image with a mean and standard deviation."""
    height, _ = tensor.shape
    std = random.randint(3, 50)
    x_offset = random.randint(0, height // 2)
    y_offset = random.randint(0, height // 2)
    gauss_image: torch.Tensor = gauss_2d_tensor_image(height, std, y_offset, x_offset, normalize=False) * weight
    threshold = gaussian(std, 0, std)
    seg_img = gauss_image > threshold
    tensor += gauss_image
    return tensor, seg_img


def add_random_gauss_blobs(tensor: torch.Tensor, n_blobs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    blob_segmentations = torch.zeros_like(tensor)
    for _ in range(n_blobs):
        tensor, blob_seg = add_random_gauss_blob(tensor)
        blob_segmentations += blob_seg
    blob_segmentations = blob_segmentations != 0
    return tensor, blob_segmentations

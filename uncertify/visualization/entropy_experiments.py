"""
Some visualization functions to understand the entropy behaviour on (synthetic) images better.
"""
from typing import Tuple
import math

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
import seaborn as sns

from uncertify.evaluation.entropy import get_entropy
from uncertify.data.utils import gauss_2d_tensor_image
from uncertify.utils.tensor_images import add_random_circles, add_random_gauss_blobs

from typing import List, Union


class ExampleFactory:
    kinds = ['flat', 'centered_gauss_blobs', 'gauss_noise', 'checkerboard', 'centered_circles']

    def __init__(self, shape: Tuple[int, int]) -> None:
        self._shape = shape
        self._grid_size = 25  # when creating multiple images and for grid arrangement, should be square number

    def create_sample(self, kind: str, normalize: bool = True) -> Union[Tensor, List[Tensor]]:
        """Creates a normed (sum=1) single image tensor (height x width) for a given kind."""
        assert kind in self.kinds, f'Provided sample type {kind} invalid, chose from {self.kinds}.'
        if kind == 'flat':
            return self._flat_distribution(normalize)
        elif kind == 'centered_gauss_blobs':
            return self._centered_gauss_blobs(normalize)
        elif kind == 'gauss_noise':
            return self._gauss_noise(normalize)
        elif kind == 'checkerboard':
            return self._checker_board(normalize)
        elif kind == 'centered_circles':
            return self._centered_circles(normalize)
        else:
            raise KeyError(f'Provided kind {kind} not supported.')

    def _flat_distribution(self, normalize: bool) -> Tensor:
        t = torch.ones(self._shape)
        if normalize:
            t /= torch.sum(t)
        return t

    def _centered_gauss_blobs(self, normalize: bool) -> List[Tensor]:
        standard_deviations = np.linspace(0.1, 100, self._grid_size)
        size, _ = self._shape
        return [gauss_2d_tensor_image(size, std, normalize=normalize) for std in standard_deviations]

    def _gauss_noise(self, normalize: bool) -> Tensor:
        image = torch.Tensor(np.random.normal(size=self._shape))
        image = torch.abs(image)
        if normalize:
            image /= torch.sum(image)
        return image

    def _checker_board(self, normalize: bool) -> Tensor:
        def checkerboard(shape):
            return np.indices(shape).sum(axis=0) % 2
        image = torch.Tensor(checkerboard(self._shape))
        if normalize:
            image /= torch.sum(image)
        return image

    def _centered_circles(self, normalize) -> List[Tensor]:
        radii = np.linspace(0.1, 100, self._grid_size)
        images = []
        for radius in radii:
            # xx and yy are 200x200 tables containing the x and y coordinates as values
            # mgrid is a mesh creation helper
            xx, yy = np.mgrid[:self._grid_size, :self._grid_size]
            # circles contains the squared distance to the (100, 100) point
            # we are just using the circle equation learnt at school
            center = {'x': self._grid_size // 2, 'y': self._grid_size // 2}
            circle = (xx - center['x']) ** 2 + (yy - center['y']) ** 2
            # donuts contains 1's and 0's organized in a donut shape
            # you apply 2 thresholds on circle to define the shape
            t = torch.FloatTensor(circle < radius ** 2)
            if normalize:
                t /= torch.sum(t)
            images.append(t)
        return images


def plot_image_and_entropy(image: torch.Tensor, mask: torch.Tensor = None) -> None:
    """Plot a single 2D torch tensor and display entropy."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if mask is None:
        mask = torch.ones_like(image).type(torch.BoolTensor)
    e = get_entropy(image, mask)
    im = ax.imshow(image, cmap='gray')
    ax.annotate(f'{e:.2f}', (10, 20), color='red', fontsize=20)
    ax.axis('off')
    plt.colorbar(im)
    plt.show()


def plot_images_and_entropy(images: List[torch.Tensor], masks: List[torch.Tensor] = None) -> None:
    """Plot a grid of images and display entropies."""
    n_images = len(images)
    assert math.isqrt(n_images), f'Number of images should be square to arrange in grid.'
    fig, axes = plt.subplots(figsize=(10, 10), nrows=int(math.sqrt(n_images)), ncols=int(math.sqrt(n_images)))
    for idx, (image, ax) in enumerate(zip(images, axes.reshape(-1))):
        mask = masks[idx] if masks is not None else torch.ones_like(image).type(torch.BoolTensor)
        e = get_entropy(image, mask)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.annotate(f'{e:.2f}', (10, 20), color='red', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_entropy_segmentations(input_batch: dict, add_circles: bool = False, add_gauss_blobs: bool = False,
                               add_steady_background: float = 0.0, normalize: bool = False, zero_out_seg: bool = False):
    fig, axes = plt.subplots(figsize=(12, 12), nrows=8, ncols=8)

    # for later correlation plot
    n_non_zeros = []
    entropies = []

    for scan, segmentation, mask, ax in zip(input_batch['scan'], input_batch['seg'], input_batch['mask'],
                                            axes.reshape(-1)):
        seg_img = segmentation[0]
        # ATTENTION: Overwriting mask!!!
        mask_img = torch.ones_like(seg_img).type(torch.BoolTensor)

        if zero_out_seg:
            seg_img = torch.zeros_like(seg_img)

        n_seg_pixels = (seg_img != 0).sum()
        if add_circles:
            seg_img = add_random_circles(seg_img, 8)
        if add_gauss_blobs:
            seg_img, blob_segs = add_random_gauss_blobs(seg_img, 1)
            n_non_zero_blobs = torch.sum(blob_segs)

        n_seg_circ_pixels = (seg_img != 0).sum() if not add_gauss_blobs else n_non_zero_blobs

        if add_steady_background > 0.0:
            seg_img += torch.ones_like(seg_img) * add_steady_background

        if normalize:
            if torch.sum(seg_img) > 0:
                seg_img = seg_img / torch.sum(seg_img)

        if not add_circles:
            use_zero_entropy = torch.sum(mask_img) < 10  # or n_seg_pixels < 10
        else:
            use_zero_entropy = n_seg_circ_pixels < 10

        if use_zero_entropy:
            entropy = 0.0
        else:
            entropy = get_entropy(seg_img, mask=mask_img)
            n_non_zeros.append(n_seg_circ_pixels)
            entropies.append(entropy)

        ax.imshow(seg_img, cmap='gray', vmin=0, vmax=0.001)  # , vmin=6e-5, vmax=6.6e-5)
        ax.annotate(f'{entropy:.2f}\n({n_seg_circ_pixels})', (10, 50), color='red', fontsize=14)
        ax.axis('off')

    plt.show()

    fig, ax = plt.subplots(figsize=(4, 8))
    sns_ax = sns.regplot(ax=ax, x=n_non_zeros, y=entropies,
                         scatter_kws={"s": 80},
                         logx=True)
    sns_ax.set_xlabel('Non-zero pixels')
    sns_ax.set_ylabel('Entropy')
    sns_ax.set_ylim([0, 1])

    plt.show()



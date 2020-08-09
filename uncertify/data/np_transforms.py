"""
The transformers in this module act on a single numpy array.
"""
from abc import ABC, abstractmethod
import logging

import numpy as np
from PIL import Image

from typing import Tuple, Dict

LOG = logging.getLogger(__name__)


class NumpyTransform(ABC):
    @staticmethod
    def check_type_with_warning(input_: np.ndarray) -> None:
        if not isinstance(input_, np.ndarray):
            raise TypeError(f'Attempting to use a numpy transform with input of type {type(input_)}. Abort.')

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class Numpy2PILTransform(NumpyTransform):
    """Transforms a Numpy nd.array into a PIL image."""
    def __call__(self, array: np.ndarray) -> Image:
        self.check_type_with_warning(array)
        return Image.fromarray(array)


class NumpyReshapeTransform(NumpyTransform):
    """Take a flattened 1D numpy array and transform into new 2D shape and returns a PIL image (for torchvision)."""

    def __init__(self, new_shape: Tuple[int, int]) -> None:
        self._new_shape = new_shape

    def __call__(self, array: np.ndarray) -> np.ndarray:
        self.check_type_with_warning(array)
        return np.reshape(array, self._new_shape)


class NumpyNormalizeTransform(NumpyTransform):
    """Normalizes a numpy array to have zero mean and unit variance.

    Note: This transformer takes NO mask into account!
    """
    def __call__(self, array: np.ndarray) -> np.ndarray:
        self.check_type_with_warning(array)
        return normalize_2d_array(array)


class NumpyNormalize01Transform(NumpyTransform):
    """Normalizes the data such that it lies in the range of [0, 1].

    Note: This transformer takes NO mask into account!
    """
    def __call__(self, array: np.ndarray) -> np.ndarray:
        self.check_type_with_warning(array)
        return (array - np.min(array)) / (np.max(array) - np.min(array))


def normalize_2d_array(input_array: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Normalizes a given input array to zero mean and unit variance. When mask is given, only consider those values."""
    assert input_array.shape == mask.shape, f'Input and mask need to have same shape: ' \
                                            f'{input_array.shape} != {mask.shape}'
    assert mask.dtype == bool, f'Mask needs to be boolean. Given: {mask.dtype}'
    if mask is not None:
        relevant_values = input_array[mask]  # gives back 1D array of values where mask has 'True' entry
    else:
        relevant_values = input_array
    mean = np.mean(relevant_values)
    std = np.std(relevant_values) + 1e-8
    return (input_array - mean) / std

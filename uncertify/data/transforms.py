import numpy as np
from PIL import Image

from typing import Tuple


class Numpy2PILTransform:
    def __call__(self, np_array: np.ndarray) -> Image:
        return Image.fromarray(np_array)


class NumpyFlat2ImgTransform:
    """Take a flattened 1D numpy array and transform into new 2D shape and returns a PIL image (for torchvision)."""
    def __init__(self, new_shape: Tuple[int, int]) -> None:
        self._new_shape = new_shape

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        return np.reshape(vector, self._new_shape)


class NumpyNormalizeTransform:
    """Normalizes a numpy array to have zero mean and unit variance."""
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return (array - array.mean()) / array.std()

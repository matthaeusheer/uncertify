import numpy as np

from typing import Tuple


class Flat2ImgTransform:
    """Take a flattened 1D numpy array and transform into new 2D shape."""
    def __init__(self, new_shape: Tuple[int, int]) -> None:
        self._new_shape = new_shape

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        return np.reshape(vector, self._new_shape)

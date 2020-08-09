import pytest
import numpy as np

from uncertify.data import np_transforms


def test_numpy_unflatten_transform():
    data = np.array([1, 2, 3, 4])
    transformed = np_transforms.NumpyReshapeTransform(new_shape=(2, 2))(data)
    target = np.array([[1, 2], [3, 4]])
    assert (transformed == target).all()


def test_numpy_normalize_transform():
    data = np.array(range(10))
    transformed = np_transforms.NumpyNormalizeTransform()(data)
    assert transformed.mean() == pytest.approx(0.0)
    assert transformed.std() == pytest.approx(1.0)


def test_numpy_normalize_01_transform():
    data = np.array(range(10))
    transformed = np_transforms.NumpyNormalize01Transform()(data)
    assert np.max(transformed) == pytest.approx(1.0)
    assert np.min(transformed) == pytest.approx(0.0)

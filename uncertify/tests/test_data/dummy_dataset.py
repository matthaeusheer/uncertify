import torch
from torch.utils.data import Dataset

from uncertify.utils.custom_types import Tensor


class DummyDataSet(Dataset):
    default_sample_shape = (1, 24, 24)

    def __init__(self, n_samples: int = 1000, sample_shape: tuple = None) -> None:
        self._n_samples = n_samples
        self._sample_shape = sample_shape if sample_shape is not None else DummyDataSet.default_sample_shape

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx) -> Tensor:
        return torch.rand(self._n_samples)

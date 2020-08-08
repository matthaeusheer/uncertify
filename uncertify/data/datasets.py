from pathlib import Path

import h5py
from cached_property import cached_property
from torch.utils.data import Dataset

from uncertify.models.custom_types import Tensor

from typing import Any, Tuple


ALLOWED_SUBSETS = ['train', 'val', 'test']


class Brats2017HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_path: Path, transform: Any = None) -> None:
        self._h5py_file = h5py.File(hdf5_file_path, 'r')
        self._transform = transform

    @cached_property
    def dataset_shape(self) -> Tuple[int, int]:
        num_samples, flat_dimensions = self._h5py_file['Scan'].shape
        return num_samples, flat_dimensions

    def __len__(self) -> int:
        return self.dataset_shape[0]

    def __getitem__(self, idx) -> Tensor:
        scan_sample = self._h5py_file['Scan'][idx]
        seg_sample = self._h5py_file['Seg'][idx]
        if self._transform is not None:
            scan_sample = self._transform(scan_sample)
            seg_sample = self._transform(seg_sample)
        return {'scan': scan_sample, 'seg': seg_sample}

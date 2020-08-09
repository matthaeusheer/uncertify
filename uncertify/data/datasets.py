from pathlib import Path

import h5py
from cached_property import cached_property
from torch.utils.data import Dataset

from typing import Any, Tuple


class HDF5Dataset(Dataset):
    """Serves as a base class for BraTS2017 and CamCan datasets which come in HDF5 format."""

    def __init__(self, hdf5_file_path: Path, transform: Any = None) -> None:
        self._h5py_file = h5py.File(hdf5_file_path, 'r')
        self._transform = transform

    @cached_property
    def dataset_shape(self) -> Tuple[int, int]:
        num_samples, flat_dimensions = self._h5py_file['Scan'].shape
        return num_samples, flat_dimensions

    def __len__(self) -> int:
        return self.dataset_shape[0]


class Brats2017HDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx) -> dict:
        scan_sample = self._h5py_file['Scan'][idx]
        seg_sample = self._h5py_file['Seg'][idx]
        if self._transform is not None:
            scan_sample = self._transform(scan_sample)
            seg_sample = self._transform(seg_sample)
        return {'scan': scan_sample, 'seg': seg_sample}


class CamCanHDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx) -> dict:
        scan_sample = self._h5py_file['Scan'][idx]
        if self._transform is not None:
            scan_sample = self._transform(scan_sample)
        return {'scan': scan_sample}

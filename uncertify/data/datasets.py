from pathlib import Path

import h5py
import torchvision
from cached_property import cached_property
from torch.utils.data import Dataset

from uncertify.data.np_transforms import NumpyTransform
from uncertify.data.dict_transforms import DictSampleTransform

from typing import Any, Tuple


class HDF5Dataset(Dataset):
    """Serves as a base class for BraTS2017 and CamCan datasets which come in HDF5 format."""

    def __init__(self, hdf5_file_path: Path, transform: torchvision.transforms.Compose = None) -> None:
        if transform is not None:
            assert isinstance(transform, torchvision.transforms.Compose), f'Only Compose transform allowed.'
        self._h5py_file = h5py.File(hdf5_file_path, 'r')
        self._transform = transform

    @cached_property
    def dataset_shape(self) -> Tuple[int, int]:
        num_samples, flat_dimensions = self._h5py_file['Scan'].shape
        return num_samples, flat_dimensions

    def __len__(self) -> int:
        return self.dataset_shape[0]


# TODO: Combine shared functionality - dict transformation horribly inefficient.


class Brats2017HDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx) -> dict:
        scan_sample = self._h5py_file['Scan'][idx]
        seg_sample = self._h5py_file['Seg'][idx]
        mask_sample = self._h5py_file['Mask'][idx].astype(bool)
        for transform in self._transform.transforms:
            if isinstance(transform, DictSampleTransform):
                out_dict = transform({'scan': scan_sample, 'seg': seg_sample, 'mask': mask_sample})
                scan_sample = out_dict['scan']
                seg_sample = out_dict['seg']
                mask_sample = out_dict['mask']
            else:  # Assume that this transform acts on the sample directly!
                scan_sample = transform(scan_sample)
                seg_sample = transform(seg_sample)
        return {'scan': scan_sample, 'seg': seg_sample, 'mask': mask_sample}


class CamCanHDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx) -> dict:
        scan_sample = self._h5py_file['Scan'][idx]
        mask_sample = self._h5py_file['Mask'][idx].astype(bool)
        if not any([isinstance(t, DictSampleTransform) for t in self._transform.transforms]):
            mask_sample = self._transform(mask_sample)
            scan_sample = self._transform(scan_sample)
        else:
            for transform in self._transform.transforms:
                if isinstance(transform, DictSampleTransform):
                    out_dict = transform({'scan': scan_sample, 'mask': mask_sample})
                    scan_sample = out_dict['scan']
                    mask_sample = out_dict['mask']
                else:
                    scan_sample = transform(scan_sample)
                    mask_sample = transform(mask_sample)
        return {'scan': scan_sample, 'mask': mask_sample}

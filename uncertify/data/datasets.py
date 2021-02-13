from pathlib import Path

import h5py
import torch
import torchvision
from cached_property import cached_property
from torch.utils.data import Dataset

from uncertify.data.dict_transforms import DictSampleTransform
from uncertify.utils.custom_types import Tensor

from typing import Tuple


class UncertifyDataset(Dataset):
    def __init__(self, uppercase_keys: bool = False) -> None:
        self._uppercase_keys = uppercase_keys

    @property
    def _scan_key(self) -> str:
        return 'Scan' if self._uppercase_keys else 'scan'

    @property
    def _mask_key(self) -> str:
        return 'Mask' if self._uppercase_keys else 'mask'

    @property
    def _seg_key(self) -> str:
        return 'Seg' if self._uppercase_keys else 'seg'

    @property
    def name(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        """Abstract method to implement to adhere do pytorch Dataset API."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Abstract method to implement to adhere do pytorch Dataset API."""
        raise NotImplementedError


class HDF5Dataset(UncertifyDataset):
    """Serves as a base class for BraTS2017 and CamCan datasets which come in HDF5 format."""

    def __init__(self, hdf5_file_path: Path, transform: torchvision.transforms.Compose = None,
                 uppercase_keys: bool = False) -> None:
        super().__init__(uppercase_keys)
        if transform is not None:
            assert isinstance(transform, torchvision.transforms.Compose), f'Only Compose transform allowed.'
        self._hdf5_file_path = hdf5_file_path
        self._transform = transform

    @cached_property
    def dataset_shape(self) -> Tuple[int, int]:
        num_samples, flat_dimensions = h5py.File(self._hdf5_file_path, 'r')[self._scan_key].shape
        return num_samples, flat_dimensions

    @property
    def name(self) -> str:
        return self._hdf5_file_path.name

    def __len__(self) -> int:
        return self.dataset_shape[0]

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError


class Brats2017HDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx: int) -> dict:
        h5py_file = h5py.File(self._hdf5_file_path, 'r')
        scan_sample = h5py_file[self._scan_key][idx]
        seg_sample = h5py_file[self._seg_key][idx]
        mask_sample = h5py_file[self._mask_key][idx].astype(bool)
        for transform in self._transform.transforms:
            if isinstance(transform, DictSampleTransform):
                out_dict = transform({self._scan_key: scan_sample,
                                      self._seg_key: seg_sample,
                                      self._mask_key: mask_sample})
                scan_sample = out_dict[self._scan_key]
                seg_sample = out_dict[self._seg_key]
                mask_sample = out_dict[self._mask_key]
            else:  # Assume that this transform acts on the sample directly!
                scan_sample = transform(scan_sample)
                seg_sample = transform(seg_sample)
                mask_sample = transform(mask_sample)
        # mask > 0 transforms mask into bool tensor
        return {self._scan_key: scan_sample, self._seg_key: seg_sample, self._mask_key: mask_sample > 0}


class CamCanHDF5Dataset(HDF5Dataset):
    def __getitem__(self, idx: int) -> dict:
        h5py_file = h5py.File(self._hdf5_file_path, 'r')
        scan_sample = h5py_file[self._scan_key][idx]
        mask_sample = h5py_file[self._mask_key][idx].astype(bool)
        if not any([isinstance(t, DictSampleTransform) for t in self._transform.transforms]):
            mask_sample = self._transform(mask_sample)
            scan_sample = self._transform(scan_sample)
        else:
            for transform in self._transform.transforms:
                if isinstance(transform, DictSampleTransform):
                    out_dict = transform({self._scan_key: scan_sample, self._mask_key: mask_sample})
                    scan_sample = out_dict[self._scan_key]
                    mask_sample = out_dict[self._mask_key]
                else:
                    scan_sample = transform(scan_sample)
                    mask_sample = transform(mask_sample)
        return {self._scan_key: scan_sample, self._mask_key: mask_sample > 0}  # > 0 transforms mask into bool tensor


class MnistDatasetWrapper(Dataset):
    """Wrapper around MNIST to make it behave like CamCan and Brats, i.e. batch['scan'], instead of batch[0]."""
    def __init__(self, mnist_dataset: Dataset, label: int = None) -> None:
        self._mask_threshold = 0.2

        if label is not None:
            self._mnist_dataset = [item for item in mnist_dataset if item[1] == label]
        else:
            self._mnist_dataset = mnist_dataset

    def __len__(self) -> int:
        return len(self._mnist_dataset)

    def __getitem__(self, idx: int) -> dict:
        return {'scan': self._mnist_dataset[idx][0], 'label': self._mnist_dataset[idx][1],
                'mask': self._mnist_dataset[idx][0] > self._mask_threshold}  # tested by hand, works ok

    @property
    def name(self) -> str:
        return '(fashion)mnist'


class GaussianNoiseDataset(Dataset):
    def __init__(self, shape: tuple) -> None:
        """A toy dataset which is not really a dataset but simply generates images with noise on the fly."""
        self._dataset_length = 10000
        self._img_shape = shape  # e.g. (1, 128, 128)

    def __len__(self) -> int:
        return self._dataset_length

    def __getitem__(self, idx: int) -> dict:
        return {'scan': self._generate_sample()}

    def _generate_sample(self) -> Tensor:
        return torch.randn(self._img_shape)

    @property
    def name(self) -> str:
        return 'gaussian_noise'


def train_val_split(dataset: HDF5Dataset, train_fraction: float) -> Tuple[HDF5Dataset, HDF5Dataset]:
    """Performs a random split with a given fraction for training (and thus validation) set."""
    train_set_length = int(len(dataset) * train_fraction)
    val_set_length = len(dataset) - train_set_length
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_length, val_set_length])
    return train_set, val_set


def has_segmentation(dataset: HDF5Dataset) -> bool:
    """Check if the dataset has actual ground truth segmentation for lesions."""
    # NOTE: This simply checks whether the first element in the dataset has GT, not every single one.
    sample_dict = dataset.__getitem__(idx=0)
    return 'seg' in sample_dict.keys()

from enum import Enum
import logging
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader

from uncertify.data.np_transforms import NumpyReshapeTransform, Numpy2PILTransform
from uncertify.data.datasets import Brats2017HDF5Dataset, CamCanHDF5Dataset, MnistDatasetWrapper
from uncertify.data.datasets import GaussianNoiseDataset
from uncertify.data.artificial import BrainGaussBlobDataset
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Any, Optional

LOG = logging.getLogger(__name__)


class DatasetType(Enum):
    MNIST = 1
    BRATS17 = 2
    CAMCAN = 3
    GAUSS_NOISE = 4
    IBSR = 5
    CANDI = 6
    FASHION_MNIST = 7


BRATS_CAMCAN_DEFAULT_TRANSFORM = torchvision.transforms.Compose([
    NumpyReshapeTransform((200, 200)),
    Numpy2PILTransform(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()]
)

MNIST_DEFAULT_TRANSFORM = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((128, 128)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0], [1])]
)


def dataloader_factory(dataset_type: DatasetType, batch_size: int,
                       train_set_path: Path = None, val_set_path: Path = None,
                       transform: Any = None, num_workers: int = 0,
                       shuffle_train: bool = True, shuffle_val: bool = False,
                       uppercase_keys: bool = False, add_gauss_blobs: bool = False,
                       **kwargs) -> Tuple[Optional[DataLoader],
                                          Optional[DataLoader]]:
    """Returns a train and val dataloader for given dataset type based on the configuration given through arguments.

    Returns
        A tuple of (train_dataloader, val_dataloader). If there is no train or validation dataloader, e.g. BraTS is
        only used for validation, the None is set for this item, e.g. for BraTS: (None, brats_val_dataloader).
    """
    assert isinstance(dataset_type, DatasetType), f'Need to provide valid DatasetType (enum).'

    if dataset_type.value == DatasetType.MNIST.value:
        train_dataloader = mnist_train_dataloader(batch_size, shuffle_train, num_workers, transform, **kwargs)
        val_dataloader = mnist_val_dataloader(batch_size, shuffle_val, num_workers, transform, **kwargs)

    elif dataset_type.value == DatasetType.FASHION_MNIST.value:
        train_dataloader = fashion_mnist_train_dataloader(batch_size, shuffle_train, num_workers, transform,
                                                          add_gauss_blobs, **kwargs)
        val_dataloader = fashion_mnist_val_dataloader(batch_size, shuffle_val, num_workers, transform,
                                                      add_gauss_blobs, **kwargs)

    elif dataset_type.value == DatasetType.BRATS17.value:
        assert val_set_path is not None, f'For BraTS need to provide a validation dataset path!'
        train_dataloader = None
        val_dataloader = brats17_val_dataloader(val_set_path, batch_size, shuffle_val,
                                                num_workers, transform, uppercase_keys, add_gauss_blobs)

    elif dataset_type.value in [DatasetType.CAMCAN.value, DatasetType.IBSR.value, DatasetType.CANDI.value]:
        train_dataloader, val_dataloader = camcan_data_loader(train_set_path, val_set_path, batch_size,
                                                              shuffle_train, shuffle_val,
                                                              num_workers, transform, uppercase_keys, add_gauss_blobs)

    elif dataset_type.value == DatasetType.GAUSS_NOISE.value:
        noise_set = GaussianNoiseDataset(shape=(1, 128, 128))
        train_dataloader = None
        val_dataloader = DataLoader(noise_set, batch_size=batch_size)
    else:
        raise ValueError(f'DatasetType {dataset_type} not supported by this factory method.')

    return train_dataloader, val_dataloader


def brats17_val_dataloader(hdf5_path: Path, batch_size: int, shuffle: bool,
                           num_workers: int = 0, transform: Any = None,
                           uppercase_keys: bool = False, add_gauss_blobs: bool = False) -> DataLoader:
    """Create a BraTS dataloader based on a hdf_path."""
    assert hdf5_path.exists(), f'BraTS17 hdf5 file {hdf5_path} does not exist!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=hdf5_path, transform=transform,
                                             uppercase_keys=uppercase_keys)
    if add_gauss_blobs:
        brats_val_dataset = BrainGaussBlobDataset(brats_val_dataset)
    val_dataloader = DataLoader(brats_val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return val_dataloader


def camcan_data_loader(hdf5_train_path: Path = None, hdf5_val_path: Path = None,
                       batch_size: int = 64, shuffle_train: bool = True, shuffle_val: bool = False,
                       num_workers: int = 0, transform: Any = None,
                       uppercase_keys: bool = False, add_gauss_blobs: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create CamCAN train and / or val dataloaders based on paths to hdf5 files."""
    assert not all(path is None for path in {hdf5_train_path, hdf5_val_path}), \
        f'Need to give a train and / or test path!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    if hdf5_train_path is None:
        train_dataloader = None
    else:
        train_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_train_path, transform=transform,
                                      uppercase_keys=uppercase_keys)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    if hdf5_val_path is None:
        val_dataloader = None
    else:
        val_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_val_path, transform=transform, uppercase_keys=uppercase_keys)
        if add_gauss_blobs:
            val_set = BrainGaussBlobDataset(val_set)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)
    return train_dataloader, val_dataloader


def mnist_train_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0,
                           transform: Any = None, **kwargs) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                           train=True,
                                           download=True,
                                           transform=transform)
    return DataLoader(MnistDatasetWrapper(mnist_dataset=train_set, label=kwargs.get('mnist_label', None)),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def mnist_val_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0,
                         transform: Any = None, **kwargs) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    val_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                         train=False,
                                         download=True,
                                         transform=transform)
    return DataLoader(MnistDatasetWrapper(mnist_dataset=val_set, label=kwargs.get('mnist_label', None)),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def fashion_mnist_train_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0,
                                   transform: Any = None, add_gauss_blobs: bool = False, **kwargs) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(root=DATA_DIR_PATH / 'fashion_mnist_data',
                                                  train=True,
                                                  download=True,
                                                  transform=transform)
    train_set = MnistDatasetWrapper(mnist_dataset=train_set, label=kwargs.get('mnist_label', None))
    if add_gauss_blobs:
        train_set = BrainGaussBlobDataset(train_set, blob_weight=300, std_min_max=(4, 30))
    return DataLoader(train_set,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def fashion_mnist_val_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0,
                                 transform: Any = None, add_gauss_blobs: bool = False, **kwargs) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    val_set = torchvision.datasets.FashionMNIST(root=DATA_DIR_PATH / 'fashion_mnist_data',
                                                train=False,
                                                download=True,
                                                transform=transform)
    val_set = MnistDatasetWrapper(mnist_dataset=val_set, label=kwargs.get('fashion_mnist_label', None))
    if add_gauss_blobs:
        val_set = BrainGaussBlobDataset(val_set, blob_weight=300, std_min_max=(4, 30))
    return DataLoader(val_set,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def print_dataloader_info(dataloader: DataLoader, name: str) -> None:
    print(
        f'{name:18} dataloader: {len(dataloader):5} batches (batch_size: {dataloader.batch_size}) -> {len(dataloader) * dataloader.batch_size:10} samples.')

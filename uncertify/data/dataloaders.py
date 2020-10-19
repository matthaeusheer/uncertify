from enum import Enum
import logging
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader

from uncertify.data.np_transforms import NumpyReshapeTransform, Numpy2PILTransform
from uncertify.data.datasets import Brats2017HDF5Dataset, CamCanHDF5Dataset, MnistDatasetWrapper
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Any, Optional

LOG = logging.getLogger(__name__)


class DatasetType(Enum):
    MNIST = 1
    BRATS17 = 2
    CAMCAN = 3


BRATS_CAMCAN_DEFAULT_TRANSFORM = torchvision.transforms.Compose([
    NumpyReshapeTransform((200, 200)),
    Numpy2PILTransform(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()
])


def dataloader_factory(dataset_type: DatasetType, batch_size: int, path: Path = None,
                       transform: Any = None, num_workers: int = 0,
                       shuffle_train: bool = True, shuffle_val: bool = False,
                       uppercase_keys: bool = False) -> Tuple[Optional[DataLoader],
                                                              Optional[DataLoader]]:
    """Returns a train and val dataloader for given dataset type."""
    assert isinstance(dataset_type, DatasetType), f'Need to provide valid DatasetType (enum).'
    if dataset_type is DatasetType.MNIST:
        return (mnist_train_dataloader(batch_size, shuffle_train, num_workers, transform),
                mnist_val_dataloader(batch_size, num_workers, transform))
    elif dataset_type is DatasetType.BRATS17:
        assert path is not None, f'BRATS17 needs a path in the factory!'
        return None, brats17_val_dataloader(path, batch_size, shuffle_val, num_workers, transform, uppercase_keys)
    elif dataset_type is DatasetType.CAMCAN:
        assert path is not None, f'CamCAN needs a path in the factory!'
        return camcan_data_loader(path, batch_size, shuffle_train, num_workers, transform, uppercase_keys), None
    else:
        raise ValueError(f'DatasetType {dataset_type} not supported by this factory method.')


def brats17_val_dataloader(hdf5_path: Path, batch_size: int, shuffle: bool,
                           num_workers: int = 0, transform: Any = None,
                           uppercase_keys: bool = False) -> DataLoader:
    """Create a BraTS dataloader based on a hdf_path."""
    assert hdf5_path.exists(), f'BraTS17 hdf5 file {hdf5_path} does not exist!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=hdf5_path, transform=transform,
                                             uppercase_keys=uppercase_keys)
    return DataLoader(brats_val_dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def camcan_data_loader(hdf5_path: Path, batch_size: int, shuffle: bool,
                       num_workers: int = 0, transform: Any = None,
                       uppercase_keys: bool = False) -> DataLoader:
    """Create a CamCAN dataloader based on a hdf_path."""
    assert hdf5_path.exists(), f'CamCAN hdf5 file {hdf5_path} does not exist!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    camcan_dataset = CamCanHDF5Dataset(hdf5_file_path=hdf5_path, transform=transform,
                                       uppercase_keys=uppercase_keys)
    camcan_dataloader = DataLoader(camcan_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers)
    return camcan_dataloader


def mnist_train_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0, transform: Any = None) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                           train=True,
                                           download=True,
                                           transform=transform)
    return DataLoader(MnistDatasetWrapper(mnist_dataset=train_set),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)


def mnist_val_dataloader(batch_size: int, num_workers: int = 0, transform: Any = None) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    val_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                         train=False,
                                         download=True,
                                         transform=transform)
    return DataLoader(MnistDatasetWrapper(mnist_dataset=val_set),
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)

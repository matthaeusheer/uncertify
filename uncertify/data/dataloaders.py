from enum import Enum

import torchvision
from torch.utils.data import DataLoader

from uncertify.data.transforms import NumpyFlat2ImgTransform, Numpy2PILTransform
from uncertify.data.datasets import Brats2017HDF5Dataset, CamCanHDF5Dataset
from uncertify.common import DATA_DIR_PATH

from typing import Tuple, Any, Optional


class DatasetType(Enum):
    MNIST = 1
    BRATS17 = 2
    CAMCAN = 3


def dataloader_factory(dataset_type: DatasetType, batch_size: int, transform: Any = None, num_workers: int = 0,
                       shuffle_train: bool = True) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Returns a train and val dataloader for given dataset type."""
    assert isinstance(dataset_type, DatasetType), f'Need to provide valid DatasetType (enum).'
    if dataset_type == DatasetType.MNIST:
        return (mnist_train_dataloader(batch_size, shuffle_train, num_workers, transform),
                mnist_val_dataloader(batch_size, num_workers, transform))
    elif dataset_type == DatasetType.BRATS17:
        return (None,
                brats17_val_dataloader(batch_size, False, num_workers, transform))
    elif dataset_type == DatasetType.CAMCAN:
        return (camcan_data_loader('train', batch_size, shuffle_train, num_workers, transform),
                camcan_data_loader('val', batch_size, False, num_workers, transform))
    else:
        raise ValueError(f'DatasetType {dataset_type} not supported by this factory method.')


def brats17_val_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0, transform: Any = None) -> DataLoader:
    default_location = DATA_DIR_PATH / 'brats/brats_all_val.hdf5'
    assert default_location.exists(), f'Default BraTS17 hdf5 file {default_location} does not exist!'
    if transform is None:
        transform = torchvision.transforms.Compose([NumpyFlat2ImgTransform(new_shape=(200, 200)),
                                                    torchvision.transforms.ToTensor()])
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=default_location, transform=transform)
    return DataLoader(brats_val_dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def camcan_data_loader(mode: str, batch_size: int, shuffle_train: bool,
                       num_workers: int = 0, transform: Any = None) -> DataLoader:
    assert mode in {'train', 'val'}, f'Wrong mode.'
    if mode == 'val':
        default_location = DATA_DIR_PATH / 'camcan/camcan_t2_val_set.hdf5'
    elif mode == 'train':
        default_location = DATA_DIR_PATH / 'camcan/camcan_t2_train_set.hdf5'
    else:
        raise ValueError(f'Cannot handle mode {mode}.')
    assert default_location.exists(), f'Default camcan hdf5 file {default_location} does not exist!'
    if transform is None:  # default transform
        transform = torchvision.transforms.Compose([NumpyFlat2ImgTransform(new_shape=(200, 200)),
                                                    Numpy2PILTransform(),
                                                    torchvision.transforms.Resize((128, 128)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize([0.0], [1.0])])
    camcan_dataset = CamCanHDF5Dataset(hdf5_file_path=default_location, transform=transform)
    camcan_dataloader = DataLoader(camcan_dataset, batch_size=batch_size,
                                   shuffle=shuffle_train if mode == 'train' else False, num_workers=num_workers)
    return camcan_dataloader


def mnist_train_dataloader(batch_size: int, shuffle: bool, num_workers: int = 0, transform: Any = None) -> DataLoader:
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                    torchvision.transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                           train=True,
                                           download=True,
                                           transform=transform)
    return DataLoader(train_set,
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
    return DataLoader(val_set,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers)

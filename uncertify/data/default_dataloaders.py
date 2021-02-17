"""
A collection of default dataloaders, often used together throughout this project. That way they don't have
to be re-defined in every notebook to keep stuff consistent.
"""
from dataclasses import dataclass
from pathlib import Path
import logging

import torchvision
from torch.utils.data import DataLoader

from uncertify.data.dataloaders import dataloader_factory, DatasetType
from uncertify.data.dataloaders import print_dataloader_info
from uncertify.data.transforms import H_FLIP_TRANSFORM, V_FLIP_TRANSFORM
from uncertify.data.datasets import GaussianNoiseDataset
from uncertify.data.dataloaders import MNIST_DEFAULT_TRANSFORM
from uncertify.common import DATA_DIR_PATH, HD_DATA_PATH

from typing import Dict, List

LOG = logging.getLogger(__name__)


@dataclass
class DefaultDataloaderParams:
    batch_size: int
    num_workers: int
    shuffle_val: bool


@dataclass
class DefaultDatasetPaths:
    brats_t2_path: Path = DATA_DIR_PATH / 'processed/brats17_t2_bc_std_bv3.5.hdf5'
    brats_t2_hm_path: Path = HD_DATA_PATH / 'processed/brats17_t2_hm_bc_std_bv3.5.hdf5'
    brats_t1_path: Path = DATA_DIR_PATH / 'processed/brats17_t1_bc_std_bv3.5.hdf5'
    brats_t1_hm_path: Path = HD_DATA_PATH / 'processed/brats17_t1_hm_bc_std_bv3.5.hdf5'
    camcan_t2_val_path: Path = DATA_DIR_PATH / 'processed/camcan_val_t2_hm_std_bv3.5_xe.hdf5'
    camcan_t2_train_path: Path = DATA_DIR_PATH / 'processed/camcan_train_t2_hm_std_bv3.5_xe.hdf5'
    ibsr_t1_train_path: Path = HD_DATA_PATH / 'processed/ibsr_train_t1_std_bv3.5_l10_xe.hdf5'
    ibsr_t1_val_path: Path = HD_DATA_PATH / 'processed/ibsr_val_t1_std_bv3.5_l10_xe.hdf5'
    candi_t1_train_path: Path = HD_DATA_PATH / 'processed/candi_train_t1_std_bv3.5_l10_xe.hdf5'
    candi_t1_val_path: Path = HD_DATA_PATH / 'processed/candi_val_t1_std_bv3.5_l10_xe.hdf5'


def default_dataloader_dict_factory(batch_size: int = 155, num_workers: int = 12,
                                    shuffle_val: bool = False) -> Dict[str, DataLoader]:
    """Returns a dictionary with dataloader names as keys and the respective dataloader as values."""

    params = DefaultDataloaderParams(batch_size, num_workers, shuffle_val)
    paths = DefaultDatasetPaths()

    _, brats_val_t2_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                    val_set_path=paths.brats_t2_path,
                                                    shuffle_val=params.shuffle_val, num_workers=params.num_workers)
    _, brats_val_t1_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                    val_set_path=paths.brats_t1_path,
                                                    shuffle_val=params.shuffle_val, num_workers=params.num_workers)
    _, brats_val_t2_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                       val_set_path=paths.brats_t2_hm_path,
                                                       shuffle_val=params.shuffle_val,
                                                       num_workers=params.num_workers)
    _, brats_val_t1_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                       val_set_path=paths.brats_t1_hm_path,
                                                       shuffle_val=params.shuffle_val,
                                                       num_workers=params.num_workers)

    _, brats_val_t2_hflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                          val_set_path=paths.brats_t2_path,
                                                          shuffle_val=params.shuffle_val,
                                                          num_workers=params.num_workers,
                                                          transform=H_FLIP_TRANSFORM)

    _, brats_val_t2_vflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=params.batch_size,
                                                          val_set_path=paths.brats_t2_path,
                                                          shuffle_val=params.shuffle_val,
                                                          num_workers=params.num_workers,
                                                          transform=V_FLIP_TRANSFORM)

    camcan_train_dataloader, camcan_val_dataloader = dataloader_factory(DatasetType.CAMCAN,
                                                                        batch_size=params.batch_size,
                                                                        val_set_path=paths.camcan_t2_val_path,
                                                                        train_set_path=paths.camcan_t2_train_path,
                                                                        shuffle_val=params.shuffle_val,
                                                                        shuffle_train=True,
                                                                        num_workers=params.num_workers)

    camcan_lesional_train_dataloader, camcan_lesional_val_dataloader = dataloader_factory(DatasetType.CAMCAN,
                                                                                          batch_size=params.batch_size,
                                                                                          val_set_path=paths.camcan_t2_val_path,
                                                                                          train_set_path=paths.camcan_t2_train_path,
                                                                                          shuffle_val=params.shuffle_val,
                                                                                          shuffle_train=True,
                                                                                          num_workers=params.num_workers,
                                                                                          add_gauss_blobs=True)

    ibsr_train_t1_dataloader, ibsr_val_t1_dataloader = dataloader_factory(DatasetType.IBSR,
                                                                          batch_size=params.batch_size,
                                                                          val_set_path=paths.ibsr_t1_val_path,
                                                                          train_set_path=paths.ibsr_t1_train_path,
                                                                          shuffle_val=params.shuffle_val,
                                                                          shuffle_train=True,
                                                                          num_workers=params.num_workers)

    candi_train_t1_dataloader, candi_val_t1_dataloader = dataloader_factory(DatasetType.CANDI,
                                                                            batch_size=params.batch_size,
                                                                            val_set_path=paths.candi_t1_val_path,
                                                                            train_set_path=paths.candi_t1_train_path,
                                                                            shuffle_val=params.shuffle_val,
                                                                            shuffle_train=True,
                                                                            num_workers=params.num_workers)

    noise_dataloader = DataLoader(GaussianNoiseDataset(shape=(1, 128, 128)), batch_size=params.batch_size)

    _, mnist_val_dataloader = dataloader_factory(DatasetType.MNIST, batch_size=params.batch_size,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Resize((128, 128)),
                                                     torchvision.transforms.ToTensor()
                                                 ]))
    fashion_mnist_train_dataloader, fashion_mnist_val_dataloader = dataloader_factory(
        DatasetType.FASHION_MNIST,
        batch_size=params.batch_size,
        transform=MNIST_DEFAULT_TRANSFORM)

    fashion_mnist_art_train_dataloader, fashion_mnist_art_val_dataloader = dataloader_factory(
        DatasetType.FASHION_MNIST,
        batch_size=params.batch_size,
        transform=MNIST_DEFAULT_TRANSFORM,
        add_gauss_blobs=True)

    dataloader_dict = {
        'CamCAN T2': camcan_train_dataloader,
        # 'CamCAN T2 val': camcan_val_dataloader,
        'CamCAN T2 lesion': camcan_lesional_val_dataloader,
        # 'CamCAN lesion val': camcan_lesional_val_dataloader,
        'BraTS T2': brats_val_t2_dataloader,
        'BraTS T1': brats_val_t1_dataloader,
        'BraTS T2 HM': brats_val_t2_hm_dataloader,
        'BraTS T1 HM': brats_val_t1_hm_dataloader,
        'Gaussian noise': noise_dataloader,
        'MNIST': mnist_val_dataloader,
        'FashionMNIST': fashion_mnist_train_dataloader,
        'FashionMNIST lesion': fashion_mnist_art_train_dataloader,
        'BraTS T2 HFlip': brats_val_t2_hflip_dataloader,
        'BraTS T2 VFlip': brats_val_t2_vflip_dataloader,
        'IBSR T1 train': ibsr_train_t1_dataloader,
        'IBSR T1 val': ibsr_val_t1_dataloader,
        'CANDI T1 train': candi_train_t1_dataloader,
        'CANDI T1 val': candi_val_t1_dataloader,
    }
    LOG.info(f'Created {len(dataloader_dict)} default dataloaders (batch_size: {batch_size}):')
    LOG.info(f'{" | ".join(dataloader_dict.keys())}')
    print_dataloader_dict(dataloader_dict)
    return dataloader_dict


def filter_dataloader_dict(dataloader_dict: dict, contains: List[str] = None, exclude: List[str] = None) -> dict:
    """Utility function to easily select or exclude items from the dataloader dict."""
    if contains is not None:
        assert isinstance(contains, list)
        if len(contains) > 0:
            filtered_dict = {}

            for name, loader in dataloader_dict.items():
                add_ok = True
                for item in contains:
                    if item not in name:
                        add_ok = False
                if add_ok:
                    filtered_dict[name] = loader
            dataloader_dict = filtered_dict

    if exclude is not None:
        assert isinstance(exclude, list)
        if len(exclude) > 0:
            filtered_dict = {}
            for name, loader in dataloader_dict.items():
                add_ok = True
                for item in exclude:
                    if item in name:
                        add_ok = False
                if add_ok:
                    filtered_dict[name] = loader
            dataloader_dict = filtered_dict
    return dataloader_dict


def print_dataloader_dict(dataloader_dict: dict) -> None:
    """Simply print some information about this dataloader dict."""
    print(10 * '- ')
    for name, dataloader in dataloader_dict.items():
        print_dataloader_info(dataloader, name)
    print(10 * '- ')

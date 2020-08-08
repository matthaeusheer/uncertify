"""
This script servers as a template for all subsequent scripts.
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.transforms import Compose

import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.models.vae import VariationalAutoEncoder
from uncertify.models.vae_baur2020 import BaurEncoder, BaurDecoder
from uncertify.models.vae_adaptive_cnn import Encoder, Decoder
from uncertify.data.dataloaders import dataloader_factory, DatasetType
from uncertify.data.transforms import Numpy2PILTransform, NumpyFlat2ImgTransform, NumpyNormalizeTransform
from uncertify.log import setup_logging
from uncertify.common import DATA_DIR_PATH

from typing import Tuple


def parse_args() -> argparse.Namespace:
    """Use argparse to parse command line arguments and pass it on to the main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """"Main entry point for our program."""

    logger = TensorBoardLogger(str(DATA_DIR_PATH / 'lightning_logs'), name=Path(__file__).stem)

    trainer_kwargs = {'logger': logger,
                      'default_root_dir': str(DATA_DIR_PATH / 'lightning_logs'),
                      # 'max_epochs': 20,
                      'val_check_interval': 0.2,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      # 'limit_train_batches': 0.2,
                      # 'limit_val_batches': 0.5,
                      'fast_dev_run': True}
    trainer = pl.Trainer(**trainer_kwargs)
    camcan_brats_transform = Compose([NumpyFlat2ImgTransform(new_shape=(200, 200)),
                                      NumpyNormalizeTransform(),
                                      Numpy2PILTransform(),
                                      torchvision.transforms.Resize((128, 128)),
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(mean=[0.0], std=[1.0])])
    mnist_transform = Compose([torchvision.transforms.Resize((128, 128)),
                               torchvision.transforms.ToTensor()])
    train_dataloader, val_dataloader = dataloader_factory(DatasetType.CAMCAN, batch_size=64,
                                                          transform=camcan_brats_transform)
    model = VariationalAutoEncoder(BaurEncoder(), BaurDecoder(), get_batch_fn=lambda x: x['scan'])
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    setup_logging()
    main(parse_args())

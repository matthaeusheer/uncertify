"""
This script servers as a template for all subsequent scripts.
"""
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.models.variational_auto_encoder import VariationalAutoEncoder
from uncertify.models.baur_vae import BaurEncoder, BaurDecoder
from uncertify.log import setup_logging
from uncertify.common import DATA_DIR_PATH


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
    model = VariationalAutoEncoder(BaurEncoder(), BaurDecoder())

    logger = TensorBoardLogger(str(DATA_DIR_PATH / 'lightning_logs'), name='baur_vae')

    trainer_kwargs = {'logger': logger,
                      'max_epochs': 20,
                      'val_check_interval': 0.25,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      # 'precision': 16,  # needs apex installed
                      'train_percent_check': 10,
                      'fast_dev_run': False}
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model)


if __name__ == '__main__':
    setup_logging()
    main(parse_args())

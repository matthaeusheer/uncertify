"""
This script servers as a template for all subsequent scripts.
"""
import argparse

from apex import amp
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.models.variational_auto_encoder import VariationalAutoEncoder
from uncertify.models.model_factory import vanilla_encoder_decoder_factory
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
    amp.register_float_function(torch, 'sigmoid')
    latent_dims = 100
    hidden_conv_channels = [32, 64, 128, 265, 512]
    input_channels = 1  # depends on dataset, greyscale: 1, RGB: 3
    flat_conv_output_dim = 512  # run and check the dimension for the settings above

    logger = TensorBoardLogger(str(DATA_DIR_PATH / 'lightning_logs'), name='vanilla_cnn_vae')

    trainer_kwargs = {'logger': logger,
                      'default_root_dir': str(DATA_DIR_PATH / 'lightning_logs'),
                      'max_epochs': 20,
                      'val_check_interval': 0.25,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      'limit_train_batches': 0.1,
                      'limit_val_batches': 0.1,
                      'fast_dev_run': False}
    trainer = pl.Trainer(**trainer_kwargs)
    encoder, decoder = vanilla_encoder_decoder_factory(latent_dims=latent_dims,
                                                       in_channels=input_channels,
                                                       hidden_conv_channels=hidden_conv_channels,
                                                       flat_conv_output_dim=flat_conv_output_dim)

    model = VariationalAutoEncoder(encoder, decoder)
    trainer.fit(model)


if __name__ == '__main__':
    setup_logging()
    main(parse_args())

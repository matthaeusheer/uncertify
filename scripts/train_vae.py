import argparse
from pathlib import Path
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchvision
from torchvision.transforms.transforms import Compose

import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.models.vae import VariationalAutoEncoder
from uncertify.models.simple_vae import SimpleVariationalAutoEncoder
from uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.data.dataloaders import dataloader_factory, DatasetType
from uncertify.data.np_transforms import Numpy2PILTransform, NumpyReshapeTransform
from uncertify.log import setup_logging
from uncertify.models.beta_annealing import beta_config_factory
from utils import ArgumentParserWithDefaults
from uncertify.common import DATA_DIR_PATH

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Use argparse to parse command line arguments and pass it on to the main function."""
    parser = ArgumentParserWithDefaults()
    parser.add_argument(
        '-d',
        '--dataset',
        choices=['mnist', 'camcan'],
        default='camcan',
        help='Which dataset to use for training.')
    parser.add_argument(
        '-w',
        '--num-workers',
        type=int,
        default=0,
        help='How many workers to use for data loading.'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training.'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='vae',
        choices=['vae', 'simple_vae'],
        help='Which version of VAE to train.'
    )
    parser.add_argument(
        '-a',
        '--annealing',
        type=str,
        default='monotonic',
        choices=['constant', 'monotonic', 'cyclic', 'sigmoid'],
        help='Which beta annealing strategy to choose.'
    )
    parser.add_argument(
        '--beta-final',
        type=float,
        default=1.0,
        help='Beta (KL) weight if constant of final value if monotonic or cyclic annealing.'
    )
    parser.add_argument(
        '--beta-start',
        type=float,
        default=0.0,
        help='Beta (KL weight) start / low value if monotonic or cyclic annealing.'
    )
    parser.add_argument(
        '--final-train-step',
        type=int,
        default=2000,
        help='Training step where final beta value is reached when doing monotonic annealing.'
    )
    parser.add_argument(
        '--cycle-size',
        type=int,
        default=1000,
        help='Cycle size in train steps.'
    )
    parser.add_argument(
        '--cycle-size-const-fraction',
        type=int,
        default=0.5,
        help='Fraction of steps during one cycle which is constant.'
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    LOG.info(f'Argparse args: {args.__dict__}')
    logger = TensorBoardLogger(str(DATA_DIR_PATH / 'lightning_logs'), name=Path(__file__).stem)
    trainer_kwargs = {'logger': logger,
                      'default_root_dir': str(DATA_DIR_PATH / 'lightning_logs'),
                      'val_check_interval': 0.5,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      'distributed_backend': 'ddp',
                      #'limit_train_batches': 0.1,
                      #'limit_val_batches': 0.1,
                      'profiler': True,
                      'fast_dev_run': False}
    early_stop_callback = EarlyStopping(
        monitor='avg_val_mean_total_loss',
        min_delta=0.001,
        patience=5,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        verbose=True,
        monitor='avg_val_mean_total_loss',
        mode='min'
    )
    trainer = pl.Trainer(**trainer_kwargs, checkpoint_callback=checkpoint_callback)   #, early_stop_callback=early_stop_callback)

    if args.dataset == 'mnist':
        transform = Compose([torchvision.transforms.Resize((128, 128)),
                             torchvision.transforms.ToTensor()])
        dataset_type = DatasetType.MNIST

    elif args.dataset == 'camcan':
        transform = Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        dataset_type = DatasetType.CAMCAN

    else:
        raise ValueError(f'Dataset arg "{args.dataset}" not supported.')

    train_dataloader, val_dataloader = dataloader_factory(dataset_type,
                                                          batch_size=args.batch_size,
                                                          transform=transform,
                                                          num_workers=args.num_workers)
    beta_config = beta_config_factory(args.annealing, args.beta_final, args.beta_start,
                                      args.final_train_step, args.cycle_size, args.cycle_size_const_fraction)
    if args.model == 'vae':
        n_m_factor = 1.0  # len(train_dataloader.dataset) / train_dataloader.batch_size
        model = VariationalAutoEncoder(encoder=BaurEncoder(), decoder=BaurDecoder(),
                                       beta_config=beta_config,
                                       n_m_factor=n_m_factor)
    elif args.model == 'simple_vae':
        model = SimpleVariationalAutoEncoder(BaurEncoder(), BaurDecoder())
    else:
        raise ValueError(f'Unrecognized model version {args.model}')
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    setup_logging()
    main(parse_args())

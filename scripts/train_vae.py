import os
import argparse
from pathlib import Path
import logging
from pprint import pformat

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchvision
from torchvision.transforms.transforms import Compose

import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.models.vae import VariationalAutoEncoder
from uncertify.models.simple_vae import SimpleVariationalAutoEncoder
from uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.models.zimmerer import ZimmererEncoder, ZimmererDecoder
from uncertify.data.dataloaders import dataloader_factory, DatasetType
from uncertify.data.dataloaders import BRATS_CAMCAN_DEFAULT_TRANSFORM, MNIST_DEFAULT_TRANSFORM
from uncertify.data.np_transforms import Numpy2PILTransform, NumpyReshapeTransform
from uncertify.training.lightning_callbacks import SaveHyperParamsCallback
from uncertify.log import setup_logging
from uncertify.io.json import store_dict
from uncertify.models.beta_annealing import beta_config_factory
from utils import ArgumentParserWithDefaults
from uncertify.common import DATA_DIR_PATH

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Use argparse to parse command line arguments and pass it on to the main function."""
    parser = ArgumentParserWithDefaults()
    parser.add_argument(
        '-w',
        '--num-workers',
        type=int,
        default=0,
        help='How many workers to use for data loading.'
    )
    parser.add_argument(
        '-d',
        '--dataset',
        choices=['mnist', 'camcan', 'fashion', 'brats'],
        default='camcan',
        help='Which dataset to use for training.')
    parser.add_argument(
        '--train-set-path',
        type=Path,
        default=None,
        help='Path to HDF5 file holding training data.'
    )
    parser.add_argument(
        '--val-set-path',
        type=Path,
        default=None,
        help='Path to HDF5 file holding validation data.'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training.'
    )
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='baur',
        choices=['baur', 'zimmerer'],
        help='Which version of VAE to train.'
    )
    parser.add_argument(
        '-a',
        '--annealing',
        type=str,
        default='monotonic',
        choices=['constant', 'monotonic', 'cyclic', 'sigmoid', 'decay'],
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
    parser.add_argument(
        '--ood-set-paths',
        nargs='*',
        type=Path,
        help='Paths to HDF5 files holding OOD datasets which will be reconstructed during training.'
    )
    parser.add_argument(
        '--out-dir-path',
        type=Path,
        default=DATA_DIR_PATH,
        help='Output directory, logs (tensorboard, models, ...) will be stored here in a lightning_logs folder.'
    )
    parser.add_argument(
        '--log-dir-name',
        default=Path(__file__).stem,
        help='The folder name within the "lightning_logs" directory in which to store the runs.'
    )
    parser.add_argument(
        '--max-n-epochs',
        type=int,
        default=100,
        help='The maximum number of epochs to train for.'
    )
    parser.add_argument(
        '--n-ensembles',
        type=int,
        default=1,
        help='Number of (ensemble) models to train with the same parameter settings.'
    )
    parser.add_argument(
        '--fast-dev-run',
        default=False,
        action='store_true',
        help='Runs a quick train and validation loop for debugging.'
    )
    parser.add_argument(
        '--mask-start-step',
        type=int,
        default=0,
        help='After this amount of training steps enable masking in train steps.'
    )
    return parser.parse_args()


def get_callbacks() -> list:
    """Define all pytorch lightning callbacks to be passed into the lightning trainer."""
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        verbose=True,
        monitor='avg_val_mean_total_loss',
        mode='min'
    )
    callbacks = [checkpoint_callback]
    return callbacks


def get_trainer_kwargs(args: argparse.Namespace) -> dict:
    """Define the lightning trainer keyword arguments based on the command line arguments parsed."""
    trainer_kwargs = {'default_root_dir': str(args.out_dir_path / 'lightning_logs'),
                      'val_check_interval': 0.5,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      'distributed_backend': 'ddp',
                      # 'limit_train_batches': 0.2,
                      # 'limit_val_batches': 0.1,
                      'max_epochs': args.max_n_epochs,
                      'profiler': True,
                      'fast_dev_run': args.fast_dev_run}
    return trainer_kwargs


def make_args_json_serializable(args: argparse.Namespace) -> dict:
    """Replace the PosixPath items with str such that we can serialize the hyper parameters."""
    args_dict = dict(args.__dict__)
    for key, val in args_dict.items():
        if isinstance(val, Path):
            args_dict[key] = str(val)
        if isinstance(val, list):
            if any([isinstance(item, Path) for item in val]):
                args_dict[key] = [str(item) if isinstance(item, Path) else item for item in val]
    return args_dict


def main(args: argparse.Namespace) -> None:
    LOG.info(f'Argparse args: {pformat(args.__dict__)}')

    # Set up the trainer
    logger = TensorBoardLogger(str(args.out_dir_path / 'lightning_logs'), name=args.log_dir_name)
    trainer = pl.Trainer(**get_trainer_kwargs(args), logger=logger, callbacks=get_callbacks())

    os.makedirs(logger.log_dir)
    store_dict(make_args_json_serializable(args), Path(logger.log_dir), 'hyper_parameters.json')

    # Setup training and validation data
    shuffle_val = False
    if args.dataset == 'mnist':
        transform = MNIST_DEFAULT_TRANSFORM
        dataset_type = DatasetType.MNIST
    elif args.dataset == 'camcan':
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
        dataset_type = DatasetType.CAMCAN
    elif args.dataset == 'fashion':
        transform = MNIST_DEFAULT_TRANSFORM
        dataset_type = DatasetType.FASHION_MNIST
    elif args.dataset == 'brats':
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
        dataset_type = DatasetType.BRATS17
        shuffle_val = True
    else:
        raise ValueError(f'Dataset arg "{args.dataset}" not supported.')

    train_dataloader, val_dataloader = dataloader_factory(dataset_type,
                                                          batch_size=args.batch_size,
                                                          train_set_path=args.train_set_path,
                                                          val_set_path=args.val_set_path,
                                                          transform=transform,
                                                          num_workers=args.num_workers,
                                                          shuffle_train=True,
                                                          shuffle_val=shuffle_val)
    # Nasty hack
    if args.dataset == 'brats':
        train_dataloader = val_dataloader

    # Setup Beta-Annealing config
    beta_config = beta_config_factory(args.annealing, args.beta_final, args.beta_start,
                                      args.final_train_step, args.cycle_size, args.cycle_size_const_fraction)
    LOG.info(beta_config)

    # Setup OOD dataloaders to check against during training
    ood_dataloaders = []
    for hdf5_set_path in args.ood_set_paths:
        name = hdf5_set_path.name
        if 'brats' in name:
            transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
            dataset_type = DatasetType.BRATS17
        elif 'camcan' in name:
            transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
            dataset_type = DatasetType.CAMCAN
        elif 'mnist' in name:
            transform = MNIST_DEFAULT_TRANSFORM
            dataset_type = DatasetType.MNIST
        elif 'noise' in name:
            transform = None
            dataset_type = DatasetType.GAUSS_NOISE
        elif 'fashion' in name:
            transform = MNIST_DEFAULT_TRANSFORM
            dataset_type = DatasetType.FASHION_MNIST
        else:
            raise ValueError(f'OOD set name {name} not supported.')
        _, ood_val_dataloader = dataloader_factory(dataset_type,
                                                   batch_size=16,
                                                   train_set_path=None,
                                                   val_set_path=hdf5_set_path,
                                                   transform=transform,
                                                   num_workers=args.num_workers,
                                                   shuffle_val=True)
        ood_dataloaders.append({name: ood_val_dataloader})

    # Setup the actual VAE model
    n_m_factor = 1.0  # len(train_dataloader.dataset) / train_dataloader.batch_size
    if args.model == 'baur':
        model = VariationalAutoEncoder(encoder=BaurEncoder(), decoder=BaurDecoder(),
                                       beta_config=beta_config,
                                       n_m_factor=n_m_factor,
                                       ood_dataloaders=ood_dataloaders,
                                       mask_start_step=args.mask_start_step)
    elif args.model == 'zimmerer':
        model = VariationalAutoEncoder(encoder=ZimmererEncoder(), decoder=ZimmererDecoder(),
                                       beta_config=beta_config,
                                       n_m_factor=n_m_factor,
                                       ood_dataloaders=ood_dataloaders)
    elif args.model == 'simple_vae':
        model = SimpleVariationalAutoEncoder(BaurEncoder(), BaurDecoder())
    else:
        raise ValueError(f'Unrecognized model version {args.model}')
    LOG.info(model)

    # Start training - go go go!
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    setup_logging()
    cmd_args = parse_args()
    for model_idx in range(cmd_args.n_ensembles):
        LOG.info(f'Training (ensemble) model {model_idx + 1} / {cmd_args.n_ensembles}')
        main(cmd_args)

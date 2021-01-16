"""
This script servers as a template for all subsequent scripts.
"""
import argparse
from pathlib import Path
from pprint import pprint

import add_uncertify_to_path  # makes sure we can use the uncertify-ai library
import uncertify
from uncertify.log import setup_logging

from functools import partial
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

try:
    tqdm._instances.clear()
except:
    pass
import seaborn as sns

sns.set_context("poster")
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use({'figure.facecolor': 'white'})
from torch.utils.data import DataLoader

from uncertify.data.transforms import H_FLIP_TRANSFORM, V_FLIP_TRANSFORM
from uncertify.data.datasets import GaussianNoiseDataset
from uncertify.data.dataloaders import dataloader_factory, DatasetType

from uncertify.io.models import load_ensemble_models
from uncertify.utils.python_helpers import print_dict_tree, get_idx_of_closest_value

from uncertify.evaluation.ood_experiments import run_ood_evaluations, run_ood_to_ood_dict
from uncertify.evaluation.model_performance import calculate_roc, calculate_prc

from uncertify.visualization.ood_scores import plot_ood_scores, plot_most_least_ood, plot_samples_close_to_score
from uncertify.visualization.histograms import plot_multi_histogram
from uncertify.visualization.model_performance import setup_roc_prc_fig, plot_roc_curve, plot_precision_recall_curve, \
    plot_confusion_matrix
from uncertify.visualization.ood_scores import plot_ood_scores
from uncertify.visualization.entropy import plot_entropy_samples_over_range

from uncertify.common import DATA_DIR_PATH, HD_DATA_PATH


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
    batch_size = 155
    USE_N_BATCHES = 10
    NUM_WORKERS = 0
    SHUFFLE_VAL = False

    brats_t2_path = HD_DATA_PATH / 'processed/brats17_t2_bc_std_bv3.5.hdf5'
    brats_t2_hm_path = HD_DATA_PATH / 'processed/brats17_t2_hm_bc_std_bv3.5.hdf5'
    brats_t1_path = HD_DATA_PATH / 'processed/brats17_t1_bc_std_bv3.5.hdf5'
    brats_t1_hm_path = HD_DATA_PATH / 'processed/brats17_t1_hm_bc_std_bv-3.5.hdf5'
    camcan_t2_val_path = DATA_DIR_PATH / 'processed/camcan_val_t2_hm_std_bv3.5_xe.hdf5'
    camcan_t2_train_path = DATA_DIR_PATH / 'processed/camcan_train_t2_hm_std_bv3.5_xe.hdf5'
    ibsr_t1_train_path = HD_DATA_PATH / 'processed/ibsr_train_t1_std_bv3.5_l10_xe.hdf5'
    ibsr_t1_val_path = HD_DATA_PATH / 'processed/ibsr_val_t1_std_bv3.5_l10_xe.hdf5'
    candi_t1_train_path = HD_DATA_PATH / 'processed/candi_train_t1_std_bv3.5_l10_xe.hdf5'
    candi_t1_val_path = HD_DATA_PATH / 'processed/candi_val_t1_std_bv3.5_l10_xe.hdf5'

    _, brats_val_t2_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                    val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                    num_workers=NUM_WORKERS)
    _, brats_val_t1_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                    val_set_path=brats_t1_path, shuffle_val=SHUFFLE_VAL,
                                                    num_workers=NUM_WORKERS)
    _, brats_val_t2_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                       val_set_path=brats_t2_hm_path, shuffle_val=SHUFFLE_VAL,
                                                       num_workers=NUM_WORKERS)
    _, brats_val_t1_hm_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                       val_set_path=brats_t1_hm_path, shuffle_val=SHUFFLE_VAL,
                                                       num_workers=NUM_WORKERS)

    _, brats_val_t2_hflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                          val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                          num_workers=NUM_WORKERS, transform=H_FLIP_TRANSFORM)
    _, brats_val_t2_vflip_dataloader = dataloader_factory(DatasetType.BRATS17, batch_size=batch_size,
                                                          val_set_path=brats_t2_path, shuffle_val=SHUFFLE_VAL,
                                                          num_workers=NUM_WORKERS, transform=V_FLIP_TRANSFORM)

    camcan_train_dataloader, camcan_val_dataloader = dataloader_factory(DatasetType.CAMCAN, batch_size=batch_size,
                                                                        val_set_path=camcan_t2_val_path,
                                                                        train_set_path=camcan_t2_train_path,
                                                                        shuffle_val=SHUFFLE_VAL, shuffle_train=True,
                                                                        num_workers=NUM_WORKERS)
    camcan_lesional_train_dataloader, camcan_lesional_val_dataloader = dataloader_factory(DatasetType.CAMCAN,
                                                                                          batch_size=batch_size,
                                                                                          val_set_path=camcan_t2_val_path,
                                                                                          train_set_path=camcan_t2_train_path,
                                                                                          shuffle_val=False,
                                                                                          shuffle_train=True,
                                                                                          num_workers=NUM_WORKERS,
                                                                                          add_gauss_blobs=True)

    ibsr_train_dataloader, ibsr_val_dataloader = dataloader_factory(DatasetType.IBSR, batch_size=batch_size,
                                                                    val_set_path=ibsr_t1_val_path,
                                                                    train_set_path=ibsr_t1_train_path,
                                                                    shuffle_val=SHUFFLE_VAL, shuffle_train=True,
                                                                    num_workers=NUM_WORKERS)
    candi_train_dataloader, candi_val_dataloader = dataloader_factory(DatasetType.CANDI, batch_size=batch_size,
                                                                      val_set_path=candi_t1_val_path,
                                                                      train_set_path=candi_t1_train_path,
                                                                      shuffle_val=SHUFFLE_VAL, shuffle_train=True,
                                                                      num_workers=NUM_WORKERS)

    noise_set = GaussianNoiseDataset(shape=(1, 128, 128))
    noise_loader = DataLoader(noise_set, batch_size=batch_size)

    _, mnist_val_dataloader = dataloader_factory(DatasetType.MNIST, batch_size=batch_size,
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.Resize((128, 128)),
                                                     torchvision.transforms.ToTensor()
                                                 ])
                                                 )

    dataloader_dict = {'BraTS T2 val': brats_val_t2_dataloader,
                       # 'BraTS T1 val': brats_val_t1_dataloader,
                       # 'CamCAN lesion train': camcan_lesional_val_dataloader,
                       # 'CamCAN lesion val': camcan_lesional_val_dataloader,
                       # 'BraTS T2 HM val': brats_val_t2_hm_dataloader,
                       # 'BraTS T1 HM val': brats_val_t1_hm_dataloader,
                       # 'CamCAN train': camcan_train_dataloader,
                       # 'CamCAN val': camcan_val_dataloader,
                       # 'Gaussian noise': noise_loader,
                       # 'MNIST': mnist_val_dataloader,
                       # 'BraTS T2 HFlip': brats_val_t2_hflip_dataloader,
                       # 'BraTS T2 VFlip': brats_val_t2_vflip_dataloader,
                       'IBSR T1 train': ibsr_train_dataloader,
                       'CANDI T1 train': candi_train_dataloader,
                       }
    brats_dataloader_dict = {key: val for key, val in dataloader_dict.items() if 'BraTS' in key}
    # Load models
    RUN_VERSIONS = [1, 2, 3, 4, 5]
    ensemble_models = load_ensemble_models(DATA_DIR_PATH / 'masked_ensemble_models',
                                           [f'model{idx}.ckpt' for idx in RUN_VERSIONS])
    model = ensemble_models[0]

    for name, dataloader in dataloader_dict.items():
        print(f'{name:15}')

        NUM_BACTHES = 5
        OOD_METRICS = ('dose',)  # 'waic',
        DOSE_STATISTICS = ('entropy',)  # 'elbo', 'kl_div', 'rec_err')  # 'elbo', 'kl_div', 'rec_err',

        dataloader_dict = {'BraTS T2': brats_val_t2_dataloader,
                           # 'BraTS T1': brats_val_t1_dataloader,
                           # 'BraTS T2 HM': brats_val_t2_hm_dataloader,
                           # 'BraTS T1 HM': brats_val_t1_hm_dataloader,
                           'CamCAN train': camcan_train_dataloader,
                           # 'Gaussian noise': noise_loader,
                           # 'MNIST': mnist_val_dataloader,
                           'IBSR T1 train': ibsr_train_dataloader,
                           'CANDI T1 train': candi_train_dataloader,
                           }

        metrics_ood_dict = run_ood_to_ood_dict(dataloader_dict, ensemble_models, camcan_train_dataloader,
                                               num_batches=NUM_BACTHES, ood_metrics=OOD_METRICS,
                                               dose_statistics=DOSE_STATISTICS)
        print_dict_tree(metrics_ood_dict)


if __name__ == '__main__':
    setup_logging()
    main(parse_args())

from itertools import islice
import logging

from scipy import stats
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from uncertify.visualization.plotting import setup_plt_figure
from uncertify.visualization.grid import imshow_grid
from uncertify.evaluation.datasets import get_n_normal_abnormal_pixels
from uncertify.visualization.histograms import plot_multi_histogram

from typing import Tuple

LOG = logging.getLogger(__name__)


def plot_brats_batches(brats_dataloader: DataLoader, plot_n_batches: int, **kwargs) -> None:
    """Plot batches of a BraTS dataloader.

    Keyword Args:
        nrow: kwarg to change number of rows
        uppercase_keys: if True, changes 'scan' to 'Scan' to support legacy hdf5 datasets
    """
    LOG.info('Plotting BraTS2017 Dataset [scan & segmentation]')
    for sample in islice(brats_dataloader, plot_n_batches):
        nrow_kwarg = {'nrow': kwargs.get('nrow')} if 'nrow' in kwargs.keys() else dict()
        scan_key = 'Scan' if kwargs.get('uppercase_keys', False) else 'scan'
        seg_key = 'Seg' if kwargs.get('uppercase_keys', False) else 'seg'
        mask_key = 'Mask' if kwargs.get('uppercase_keys', False) else 'mask'

        mask = torch.where(sample[mask_key], sample[mask_key].type(torch.FloatTensor), -3.5 * torch.ones_like(sample[scan_key]))
        seg = torch.where(sample[seg_key].type(torch.BoolTensor), sample[seg_key].type(torch.FloatTensor), -3.5 * torch.ones_like(sample[scan_key]))
        grid = make_grid(
            torch.cat((sample[scan_key].type(torch.FloatTensor),
                       seg.type(torch.FloatTensor),
                       mask.type(torch.FloatTensor)),
                      dim=2),
            padding=0, **nrow_kwarg)
        imshow_grid(grid, one_channel=True, plt_show=True, axis='off', **kwargs)
        plt.show()


def plot_camcan_batches(camcan_dataloader: DataLoader, plot_n_batches: int, **kwargs) -> None:
    """Plot batches of a CamCAN dataloader.

    Keyword Args:
        nrow: kwarg to change number of rows
        uppercase_keys: if True, changes 'scan' to 'Scan' to support legacy hdf5 datasets
    """
    LOG.info('Plotting CamCAN Dataset [scan only]')
    nrow_kwarg = {'nrow': kwargs.get('nrow')} if 'nrow' in kwargs.keys() else dict()
    for sample in islice(camcan_dataloader, plot_n_batches):
        scan = 'Scan' if kwargs.get('uppercase_keys', False) else 'scan'
        grid = make_grid(sample[scan].type(torch.FloatTensor), padding=0, **nrow_kwarg)
        imshow_grid(grid, one_channel=True, plt_show=True, axis='off', **kwargs)
        plt.show()


def plot_samples(h5py_file: h5py.File, n_samples: int = 3, dataset_length: int = 4000, cmap: str = 'Greys_r',
                 vmin: float = None, vmax: float = None) -> None:
    """Plot samples and pixel distributions as they come out of the h5py file directly."""
    sample_indices = np.random.choice(dataset_length, n_samples)
    keys = sorted(list(h5py_file.keys()))
    for counter, idx in enumerate(sample_indices):
        fig, axes = plt.subplots(ncols=len(keys) + 1, nrows=2, figsize=(12, 12))
        mask = h5py_file['mask'][idx]
        scan = h5py_file['scan'][idx]
        masked_scan = np.where(mask.astype(bool), scan, np.zeros(scan.shape))
        min_val = np.min(masked_scan) if vmin is None else vmin
        max_val = np.max(masked_scan) if vmax is None else vmax
        masked_pixels = scan[mask.astype(bool)].flatten()
        datasets = [h5py_file[key] for key in keys] + [masked_scan]
        for dataset_name, dataset, ax in zip(keys + ['masked_scan'], datasets, np.transpose(axes)):
            if dataset_name != 'masked_scan':
                array_2d = dataset[idx]
            else:  # actually not a dataset but simply an array already
                array_2d = dataset
            im = ax[0].imshow(np.reshape(array_2d, (200, 200)), cmap=cmap, vmin=min_val, vmax=max_val)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax[0].axis('off')
            ax[0].set_title(dataset_name)
            ax[1].hist(array_2d if dataset_name != 'masked_scan' else masked_pixels, bins=30, density=False)
            try:
                description = stats.describe(array_2d if dataset_name != 'masked_scan' else masked_pixels)
            except ValueError:
                print(f'Found sample with empty mask. No statistics available.')
            else:
                ax[1].set_title(f'mean: {description.mean:.2f}, var: {description.variance:.2f}')
                print(f'{dataset_name:15}: min/max: {description.minmax[0]:.2f}/{description.minmax[1]:.2f}, '
                      f'mean: {description.mean:.2f}, variance: {description.variance:.2f}')
        plt.tight_layout()
        plt.show()


def plot_patient_histograms(dataloader: DataLoader, n_batches: int, accumulate_batches: bool = False,
                            bins: int = 40, uppercase_keys: bool = False):
    """Plot the batch-wise intensity histograms.

    Arguments
        dataloader: a hdf5 dataloader
        plot_n_batches: how many batches to take into account
        accumulate_batches: if True, stack all values from all batches and report one histogram
                            if False, do one histogram for every batch in the figure
        bins: number of bins in histograms
        uppercase_keys: if True supports legacy upper case keys
    """
    accumulated_values = []
    for idx, batch in enumerate(dataloader):
        mask = batch['mask' if not uppercase_keys else 'Mask'].cpu().detach().numpy()
        scan = batch['scan' if not uppercase_keys else 'Scan'].cpu().detach().numpy()
        masked_pixels = scan[mask != 0].flatten()
        accumulated_values.append(masked_pixels)
        if idx + 1 == n_batches:
            break

    if accumulate_batches:
        values = np.concatenate(accumulated_values)
        plot_multi_histogram(
            arrays=[values],
            plot_density=False,  # KDE
            title='Accumulated Intensities Histogram',
            xlabel='Pixel Intensity',
            hist_kwargs=dict(bins=bins),
            figsize=(12, 8),
        )
        plt.show()
    else:
        plot_multi_histogram(
            arrays=accumulated_values,
            labels=[f'Batch {idx + 1}' for idx in range(len(accumulated_values))],
            plot_density=False,  # KDE
            title='Batch-wise intensity Histograms',
            xlabel='Pixel Intensity',
            hist_kwargs=dict(bins=bins),
            figsize=(12, 8),
        )
        plt.show()


def plot_abnormal_pixel_distribution(data_loader: DataLoader, **hist_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """For a dataset with given ground truth, plot the distribution of fraction of abnormal pixels in an image.

    This is done sample-wise, i.e.

    Note: Only pixels within the brain mask are considered and only samples with abnormal pixels are considered.
    """
    normal_pixels, abnormal_pixels, total_masked_pixels = get_n_normal_abnormal_pixels(data_loader)

    fig, ax = plot_multi_histogram(
        arrays=[np.array(normal_pixels), np.array(abnormal_pixels), np.array(total_masked_pixels)],
        labels=['Normal pixels', 'Abnormal pixels', 'Mask Size'],
        plot_density=False,
        title='Distribution of the sample-wise number of normal / abnormal pixels',
        xlabel='Number of pixels',
        ylabel='Frequency',
        **hist_kwargs)
    return fig, ax


def plot_fraction_of_abnormal_pixels(data_loader: DataLoader, **hist_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """For a dataset with given ground truth, plot the distribution of fraction of abnormal pixels in an image.

    Note: Only pixels within the brain mask are considered and only samples with abnormal pixels are considered.
    """
    normal_pixels, abnormal_pixels, total_masked_pixels = get_n_normal_abnormal_pixels(data_loader)
    fractions = []
    for normal, total in zip(abnormal_pixels, total_masked_pixels):
        fraction = normal / total
        fractions.append(fraction)
    percentile_5 = np.percentile(fractions, q=5)
    fig, ax = plot_multi_histogram(
        arrays=[np.array(fractions)],
        labels=None,
        plot_density=True,
        kde_bandwidth=0.02,
        xlabel='Fraction of abnormal pixels from all pixels within brain masks',
        ylabel='Frequency',
        **hist_kwargs)
    ax.plot([percentile_5, percentile_5], [0, 3], 'g--', linewidth=2)  # TODO: Hardcoded.
    return fig, ax


def boxplot_abnormal_pixel_fraction(data_loader: DataLoader, **plt_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """A boxplot from the """
    fig, ax = setup_plt_figure(aspect='auto', **plt_kwargs)
    normal_pixels, abnormal_pixels, total_masked_pixels = get_n_normal_abnormal_pixels(data_loader)
    fractions = []
    for normal, total in zip(abnormal_pixels, total_masked_pixels):
        fraction = normal / total
        fractions.append(fraction)
    percentile_5 = np.percentile(fractions, q=5)
    ax = sns.boxplot(data=np.array(fractions), ax=ax)
    ax.set_title(f'5 percentile = {percentile_5:.2f}', fontweight='bold')
    plt.tight_layout()
    return fig, ax

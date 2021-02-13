from pathlib import Path
import itertools
from math import ceil
from typing import List

import scipy.stats
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from uncertify.visualization.grid import imshow_grid
from uncertify.utils.custom_types import Tensor
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.evaluation.utils import mask_background_to_zero, mask_background_to_value
from uncertify.utils.tensor_ops import print_scipy_stats_description
from uncertify.evaluation.inference import BatchInferenceResult
from uncertify.visualization.plotting import save_fig

from typing import Generator, Dict, Tuple

# Minimum and maximum values for 0-1 normalization for plotting
MIN_VAL = -3.5
MAX_VAL = 2


def plot_stacked_scan_reconstruction_batches(batch_generator: Generator[BatchInferenceResult, None, None],
                                             plot_n_batches: int = 3, nrow: int = 8, show_mask: bool = False,
                                             save_dir_path: Path = None, cut_tensor_to: int = None,
                                             mask_background: bool = True, **kwargs) -> None:
    """Plot the scan and reconstruction batches. Horizontally aligned are the samples from one batch.
    Vertically aligned are input image, ground truth segmentation, reconstruction, residual image, residual with
    applied threshold, ground truth and predicted segmentation in same image.
    Args:
        batch_generator: a PyTorch DataLoader as defined in the uncertify dataloaders module
        plot_n_batches: limit plotting to this amount of batches
        save_dir_path: path to directory in which to store the resulting plots - will be created if not existent
        nrow: numbers of samples in one row, default is 8
        show_mask: plots the brain mask
        cut_tensor_to: choose 8 if you need only first 8 samples but for plotting but the original tensor is way larger
        kwargs: additional keyword arguments for plotting functions
    """
    if save_dir_path is not None:
        save_dir_path.mkdir(exist_ok=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(itertools.islice(batch_generator, plot_n_batches)):
            mask = batch.mask
            max_val = torch.max(batch.scan)
            min_val = torch.min(batch.scan)

            scan = normalize_to_0_1(batch.scan, min_val, max_val)
            reconstruction = normalize_to_0_1(batch.reconstruction, min_val, max_val)
            residual = normalize_to_0_1(batch.residual)
            thresholded = batch.residuals_thresholded

            if mask_background:
                scan = mask_background_to_zero(scan, mask)
                reconstruction = mask_background_to_zero(reconstruction, mask)
                residual = mask_background_to_zero(residual, mask)
                thresholded = mask_background_to_zero(thresholded, mask)

            if batch.segmentation is not None:
                seg = mask_background_to_zero(batch.segmentation, mask)
                stacked = torch.cat((scan, seg, reconstruction, residual, thresholded), dim=2)
            else:
                stacked = torch.cat((scan, reconstruction, residual, thresholded), dim=2)
            if show_mask:
                stacked = torch.cat((stacked, mask.type(torch.FloatTensor)), dim=2)
            if cut_tensor_to is not None:
                stacked = stacked[:cut_tensor_to, ...]
            grid = torchvision.utils.make_grid(stacked, padding=0, normalize=False, nrow=nrow)
            describe = scipy.stats.describe(grid.numpy().flatten())
            print_scipy_stats_description(describe, 'normalized_grid')
            fig, ax = imshow_grid(grid, one_channel=True, vmax=1.0, vmin=0.0, plt_show=True, **kwargs)
            ax.set_axis_off()
            if save_dir_path is not None:
                img_file_name = f'batch_{batch_idx}.png'
                save_fig(fig, save_dir_path / img_file_name)


def plot_vae_output(decoder_output_batch: Tensor, nrow: int = 8, transpose_grid: bool = False,
                    **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the output tensor batch from the VAE as a grid."""
    grid = torchvision.utils.make_grid(decoder_output_batch, nrow=nrow, padding=0)
    if transpose_grid:
        grid = grid.transpose(dim0=1, dim1=2)  # swap rows and columns of the grid
    fig, ax = imshow_grid(grid, **kwargs)
    return fig, ax


def plot_pixel_value_histogram(batch_generator: Generator[Dict[str, Tensor], None, None],
                               consider_n_batches: int = 10) -> None:
    with torch.no_grad():
        for batch_idx, output_batch in enumerate(itertools.islice(batch_generator, consider_n_batches)):
            pass


# ----- DEPRECATED FUNCTIONS BELOW ----- #

def plot_vae_reconstructions(trained_model, data_loader: DataLoader,
                             device: torch.device, colormap: str = 'hot', n_batches: int = 1,
                             max_samples: int = -1, show: bool = True) -> List[plt.Figure]:
    """Run input images through VAE and visualize them against the reconstructed image.

    NOTE: This function was designed for plain pytorch models before the pytorch_lightning times!

    Args:
        trained_model: a trained pytorch model
        data_loader: a pytorch DataLoader
        device: a pytorch device
        colormap: a string representing matplotlib colormap
        n_batches: number of batches to plot
        max_samples: limits number of samples being plotted
        show: whether to call matplotlib show function
    Returns:
        a list of figures, one for each plot
    """
    plt.set_cmap(colormap)
    trained_model.eval()
    figures = []
    with torch.no_grad():
        sample_counter = 0
        for batch_features, _ in itertools.islice(data_loader, n_batches):
            batch_features = batch_features.to(device)
            out_features, _, _ = trained_model(batch_features)
            for idx, (in_feature, out_feature) in enumerate(zip(batch_features, out_features)):
                in_np = in_feature.view(28, 28).cpu().numpy()
                out_np = out_feature.view(28, 28).cpu().numpy()
                residual_np = np.abs(out_np - in_np)
                fig, (original_ax, reconstruction_ax, residual_ax) = plt.subplots(1, 3)
                original_ax.imshow(in_np, vmin=0.0, vmax=1.0)
                reconstruction_ax.imshow(out_np, vmin=0.0, vmax=1.0)
                residual_ax.imshow(residual_np, vmin=0.0, vmax=1.0)
                reconstruction_ax.set_axis_off()
                original_ax.set_axis_off()
                residual_ax.set_axis_off()
                plt.tight_layout(h_pad=0)
                figures.append(fig)
                sample_counter += 1
                if show:
                    plt.show()
                if sample_counter == max_samples:
                    break
    return figures


def plot_vae_generations(generated_samples: np.ndarray) -> plt.Figure:
    """Visualize generated samples from the variational autoencoder.

    NOTE: This function was designed for plain pytorch models before the pytorch_lightning times!

    Args:
        generated_samples: numpy version of a generated image (e.g. returned by the VEA _decode function)
    """
    images = [sample for sample in generated_samples]
    n_imgs = len(images)
    n_cols = 4
    fig, axes_2d = plt.subplots(ceil(n_imgs / n_cols), n_cols, gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(8, 8))
    for counter, img in enumerate(images):
        row_idx = counter // n_cols
        col_idx = counter % n_cols
        ax = axes_2d[row_idx][col_idx]
        ax.imshow(img)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout(w_pad=0)
    return fig

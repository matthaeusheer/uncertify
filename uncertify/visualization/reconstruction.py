import itertools
from math import ceil
from typing import List

from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from uncertify.visualization.grid import imshow_grid
from uncertify.utils.custom_types import Tensor
from uncertify.utils.tensor_ops import normalize_to_0_1


from typing import Generator, Dict


def plot_stacked_scan_reconstruction_batches(batch_generator: Generator[Dict[str, Tensor], None, None],
                                             plot_n_batches: int = 3, **kwargs) -> None:
    """Plot the scan and reconstruction batches."""
    with torch.no_grad():
        for batch in itertools.islice(batch_generator, plot_n_batches):
            scan = normalize_to_0_1(batch['scan'])
            reconstruction = normalize_to_0_1(batch['rec'])
            residual = normalize_to_0_1(batch['res'])
            if 'seg' in batch.keys():
                seg = batch['seg']
                stacked = torch.cat((scan, seg, reconstruction, residual), dim=2)
            else:
                stacked = torch.cat((scan, reconstruction, residual), dim=2)
            grid = torchvision.utils.make_grid(stacked)
            imshow_grid(grid, **kwargs)


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
    # TODO(matthaeusheer): This function is actually more general since it only plots images, rename appropriately
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

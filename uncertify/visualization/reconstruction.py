import itertools
from math import ceil
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from uncertify.tutorials.variational_auto_encoder import VariationalAutoEncoder


def plot_vae_reconstructions(trained_model: VariationalAutoEncoder, data_loader: DataLoader,
                             device: torch.device, colormap: str = 'hot', n_batches: int = 1,
                             max_samples: int = -1, show: bool = True) -> List[plt.Figure]:
    """Run input images through VAE and visualize them against the reconstructed image.
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

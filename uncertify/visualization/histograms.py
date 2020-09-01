from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity

from uncertify.data.dataloaders import DataLoader
from uncertify.visualization.plotting import setup_plt_figure
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Iterable, Generator, Dict


def plot_pixel_histogram(arrays: List[np.ndarray],
                         labels: List[str] = None,
                         plot_density: bool = True,
                         kde_bandwidth: float = 0.005, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = setup_plt_figure(aspect='auto', **kwargs)
    hist_kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=20)
    hist_kwargs.update({key: value for key, value in kwargs.items() if key in hist_kwargs})
    kde_x_values = np.linspace(min([min(arr) for arr in arrays]), max([max(arr) for arr in arrays]), 1000)
    colors = plt.cm.jet(np.linspace(0, 1, len(arrays)))
    for idx, array in enumerate(arrays):
        return_tuple = ax.hist(array, label=labels[idx] if labels is not None else None, color=colors[idx], **hist_kwargs)
        if plot_density:
            kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
            kde.fit(array.reshape(-1, 1))
            log_prob = kde.score_samples(kde_x_values[:, None])
            ax.fill_between(kde_x_values, np.exp(log_prob), alpha=0.5, color=colors[idx])
            plt.plot(array, np.full_like(array, -0.01), '|', c=colors[idx], markeredgewidth=1)
    if labels is not None:
        ax.legend()
    return fig, ax


def plot_loss_histograms(output_generators: Iterable[Generator[Dict[str, Tensor], None, None]],
                         names: Iterable[str], **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:
    """For different data loaders, plot the loss value histograms."""
    # Prepare dictionaries holding sample-wise losses for different data loaders
    kld_losses = defaultdict(list)
    rec_losses = defaultdict(list)
    # Perform inference and fill the data dicts for later plotting
    for generator, generator_name in zip(output_generators, names):
        for batch in generator:
            with torch.no_grad():
                kld_losses[generator_name].append(float(batch['kld_loss']))
                rec_losses[generator_name].append(float(batch['reconstruction_loss']))
    kld_arrays = []
    kld_labels = []
    rec_arrays = []
    rec_labels = []
    for name in names:
        kld_arrays.append(np.array(kld_losses[name]))
        kld_labels.append(f' '.join([name, 'KLD']))
        rec_arrays.append(np.array(rec_losses[name]))
        rec_labels.append(f' '.join([name, 'REC']))
    fig1, ax2 = plot_pixel_histogram(kld_arrays, kld_labels, xlabel='KLD Term Loss', **kwargs)
    fig2, ax2 = plot_pixel_histogram(rec_arrays, rec_labels, xlabel='Reconstruction Loss', **kwargs)
    return [(fig1, ax2), (fig2, ax2)]

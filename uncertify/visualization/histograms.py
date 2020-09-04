from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity

from uncertify.visualization.plotting import setup_plt_figure
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Iterable, Generator, Dict

DEFAULT_HIST_KWARGS = dict(histtype='stepfilled', alpha=0.5)


def plot_multi_histogram(arrays: List[np.ndarray],
                         labels: List[str] = None,
                         plot_density: bool = True,
                         kde_bandwidth: float = 0.005,
                         show_data_ticks: bool = False,
                         hist_kwargs: dict = None,
                         **plt_kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with a histogram consisting multiple distinct distributions.
    Args:
        arrays: each array will be converted into a plotted histogram in the plot
        labels: the labels for the corresponding arrays
        plot_density: if True, will to a kernel density estimation fit over the data
        kde_bandwidth: bandwidth for the (Gaussian) kde
        show_data_ticks: if True will show small ticks on the x-axis for each data point
        hist_kwargs: keyword arguments as dict for the plt.hist function
        plt_kwargs: keyword arguments for plot setup
    """
    fig, ax = setup_plt_figure(aspect='auto', **plt_kwargs)
    if hist_kwargs is None:
        hist_kwargs = DEFAULT_HIST_KWARGS
    else:
        DEFAULT_HIST_KWARGS.update(hist_kwargs)
        hist_kwargs = DEFAULT_HIST_KWARGS
    kde_x_values = np.linspace(min([min(arr) for arr in arrays]), max([max(arr) for arr in arrays]), 1000)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(arrays)))
    for idx, array in enumerate(arrays):
        ax.hist(array, label=labels[idx] if labels is not None else None, color=colors[idx], **hist_kwargs)
        if plot_density:
            kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
            kde.fit(array.reshape(-1, 1))
            log_prob = kde.score_samples(kde_x_values[:, None])
            ax.fill_between(kde_x_values, np.exp(log_prob), alpha=0.5, color=colors[idx])
            if show_data_ticks:
                ax.plot(array, np.full_like(array, -0.01), '|', c=colors[idx], markeredgewidth=1)
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
            has_ground_truth = 'seg' in batch.keys()
            if has_ground_truth:
                for segmentation, mask, kld, rec in zip(batch['seg'], batch['mask'],
                                                        batch['kld_loss'], batch['reconstruction_loss']):
                    n_abnormal_pixels = torch.sum(segmentation > 0)
                    n_normal_pixels = torch.sum(mask)
                    abnormal_fraction = n_abnormal_pixels / n_normal_pixels
                    is_abnormal = abnormal_fraction > 0.27  # TODO: Remove hardcoded.
                    suffix = 'abnormal' if is_abnormal else 'normal'
                    kld_losses[f'_'.join([generator_name, suffix])].append(float(batch['kld_loss']))
                    kld_losses[f'_'.join([generator_name, suffix])].append(float(batch['reconstruction_loss']))
            else:
                with torch.no_grad():
                    kld_losses[generator_name].append(float(batch['kld_loss']))
                    rec_losses[generator_name].append(float(batch['reconstruction_loss']))
    kld_arrays = []
    kld_labels = []
    rec_arrays = []
    rec_labels = []
    for name in names:
        kld_arrays.append(np.array(kld_losses[name]))
        kld_labels.append(name)
        rec_arrays.append(np.array(rec_losses[name]))
        rec_labels.append(name)
    fig1, ax2 = plot_multi_histogram(kld_arrays, kld_labels, xlabel='KLD Term Loss', **kwargs)
    fig2, ax2 = plot_multi_histogram(rec_arrays, rec_labels, xlabel='Reconstruction Loss', **kwargs)
    return [(fig1, ax2), (fig2, ax2)]

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity

from uncertify.visualization.plotting import setup_plt_figure
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Iterable, Generator, Dict, Union

DEFAULT_HIST_KWARGS = dict(histtype='stepfilled', alpha=0.5, density=True, bins=30)


def plot_multi_histogram(arrays: List[np.ndarray],
                         labels: List[str] = None,
                         plot_density: bool = True,
                         kde_bandwidth: Union[float, List[float]] = 0.005,
                         show_data_ticks: bool = True,
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
        if plt_kwargs.get('show_histograms', True):
            ax.hist(array, label=labels[idx] if labels is not None else None, color=colors[idx], **hist_kwargs)
        if plot_density:
            kde = KernelDensity(bandwidth=kde_bandwidth,
                                kernel='gaussian')
            kde.fit(array.reshape(-1, 1))
            log_prob = kde.score_samples(kde_x_values[:, None])
            ax.fill_between(kde_x_values, np.exp(log_prob), alpha=0.5, color=colors[idx],
                            label=labels[idx] if labels is not None else None)
            if show_data_ticks:
                ax.plot(array, np.full_like(array, -0.01), '|', c=colors[idx], markeredgewidth=1)
    if labels is not None:
        ax.legend(loc='best')
    return fig, ax


def plot_loss_histograms(output_generators: Iterable[Generator[Dict[str, Tensor], None, None]],
                         names: Iterable[str], **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:
    """For different data loaders, plot the sample-wise loss value histograms.

    Arguments:
        output_generators: an iterable of generators (use yield_reconstructed_batches) to create histograms for
        names: names for the respective output generators
        kwargs:
            kde_bandwidth: a list of bandwidths to use for each type of plot
    """
    # Prepare dictionaries holding sample-wise losses for different data loaders
    kl_divs = defaultdict(list)
    rec_errors = defaultdict(list)

    # Perform inference and fill the data dicts for later plotting
    for generator, generator_name in zip(output_generators, names):
        for batch in generator:
            has_ground_truth = 'seg' in batch.keys()
            if has_ground_truth:
                for segmentation, mask, kl_div, rec_err in zip(batch['seg'], batch['mask'],
                                                               batch['kl_div'], batch['rec_err']):  # sample-wise zip
                    n_abnormal_pixels = float(torch.sum(segmentation > 0))
                    n_normal_pixels = float(torch.sum(mask))
                    if n_normal_pixels == 0:
                        continue
                    abnormal_fraction = n_abnormal_pixels / n_normal_pixels
                    is_abnormal = n_abnormal_pixels > 20 # abnormal_fraction > 0.01  # TODO: Remove hardcoded.
                    suffix = 'abnormal' if is_abnormal else 'normal'
                    kl_divs[f' '.join([generator_name, suffix])].append(float(kl_div))
                    rec_errors[f' '.join([generator_name, suffix])].append(float(rec_err))
            else:
                for kl_div, rec_err in zip(batch['kl_div'], batch['rec_err']):  # sample-wise zip
                    kl_divs[generator_name].append(float(kl_div))
                    rec_errors[generator_name].append(float(rec_err))
    kld_arrays = []
    kld_labels = []
    rec_arrays = []
    rec_labels = []
    for name, arr in kl_divs.items():
        kld_arrays.append(np.array(arr))
        kld_labels.append(name)
    for name, arr in rec_errors.items():
        rec_arrays.append(np.array(rec_errors[name]))
        rec_labels.append(name)
    kde_bandwidth = kwargs.get('kde_bandwidth', None)
    kwargs.pop('kde_bandwidth')
    fig1, ax2 = plot_multi_histogram(kld_arrays, kld_labels, xlabel='KL Divergence',
                                     kde_bandwidth=kde_bandwidth[0], **kwargs)
    fig2, ax2 = plot_multi_histogram(rec_arrays, rec_labels, xlabel='Reconstruction Error',
                                     kde_bandwidth=kde_bandwidth[1], **kwargs)
    return [(fig1, ax2), (fig2, ax2)]


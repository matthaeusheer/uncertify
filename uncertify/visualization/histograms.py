from collections import defaultdict
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from uncertify.visualization.plotting import setup_plt_figure
from uncertify.evaluation.inference import BatchInferenceResult

from typing import Tuple, List, Iterable, Generator, Dict, Union

DEFAULT_HIST_KWARGS = dict(histtype='stepfilled', alpha=0.5, density=False, bins=40)


def plot_multi_histogram(arrays: List[np.ndarray],
                         labels: List[str] = None,
                         plot_density: bool = True,
                         kde_bandwidth: Union[float, List[float]] = 0.005,
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
        # Shrink current axis by 40%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  title=plt_kwargs.get('legend_title', None), frameon=False)
    return fig, ax


def plot_loss_histograms(output_generators: Iterable[Generator[BatchInferenceResult, None, None]],
                         names: Iterable[str], **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:
    """For different data loaders, plot the sample-wise loss value histograms.

    Arguments:
        output_generators: an iterable of generators (use yield_reconstructed_batches) to create histograms for
        names: names for the respective output generators
        kwargs:
            kde_bandwidth: a list of bandwidths to use for each type of plot
    """
    # Prepare dictionaries holding sample-wise losses for different data loaders (split in healthy / lesional)
    kl_divs = defaultdict(list)
    rec_errors = defaultdict(list)
    elbos = defaultdict(list)

    # Perform inference and fill the data dicts for later plotting
    for generator, generator_name in zip(output_generators, names):
        for batch in generator:
            has_ground_truth = batch.segmentation is not None
            if has_ground_truth:
                # sample-wise zip
                for segmentation, mask, kl_div, rec_err, is_lesional, is_empty in zip(batch.segmentation,
                                                                                      batch.mask,
                                                                                      batch.kl_div,
                                                                                      batch.rec_err,
                                                                                      batch.slice_wise_is_lesional,
                                                                                      batch.slice_wise_is_empty):
                    if is_empty:
                        continue
                    suffix = 'lesional' if is_lesional else 'healthy'
                    sub_generator_name = f'{generator_name} {suffix}'
                    kl_divs[sub_generator_name].append(float(kl_div))
                    rec_errors[sub_generator_name].append(float(rec_err))
                    elbos[sub_generator_name].append(float(rec_err - kl_div))
            else:
                # sample-wise zip
                for kl_div, rec_err, is_empty in zip(batch.kl_div, batch.rec_err, batch.slice_wise_is_empty):
                    if is_empty:
                        continue
                    kl_divs[generator_name].append(float(kl_div))
                    rec_errors[generator_name].append(float(rec_err))
                    elbos[generator_name].append(float(rec_err - kl_div))

    # Prepare all the arrays and corresponding labels for the histograms
    kld_arrays = []
    kld_labels = []
    rec_arrays = []
    rec_labels = []
    elbo_arrays = []
    elbo_labels = []
    for name, arr in kl_divs.items():
        if 'healthy' in name and 'lesion' in name:
            continue
        kld_arrays.append(np.array(arr))
        kld_labels.append(name)
    for name, arr in rec_errors.items():
        if 'healthy' in name and 'lesion' in name:
            continue
        rec_arrays.append(np.array(rec_errors[name]))
        rec_labels.append(name)
    for name, arr in elbos.items():
        if 'healthy' in name and 'lesion' in name:
            continue
        elbo_arrays.append(np.array(elbos[name]))
        elbo_labels.append(name)

    bandwidth_tuple = kwargs.get('kde_bandwidth', None)
    kwargs.pop('kde_bandwidth')
    kde_bandwidth, l1_elbo_bandwidth = bandwidth_tuple
    # Add labels if you want to include them in some histogram plot instead of None
    fig1, ax2 = plot_multi_histogram(kld_arrays, labels=kld_labels, xlabel='$D_{KL}$',
                                     kde_bandwidth=kde_bandwidth,
                                     **kwargs)
    fig2, ax2 = plot_multi_histogram(rec_arrays, labels=None, xlabel='$\ell_{1}$',
                                     kde_bandwidth=l1_elbo_bandwidth, xmax=2.5,
                                     **kwargs)
    fig3, ax3 = plot_multi_histogram(elbo_arrays, labels=None, xlabel='$\mathcal{L}$',
                                     kde_bandwidth=l1_elbo_bandwidth, xmax=2.5,
                                     **kwargs)
    return [(fig1, ax2), (fig2, ax2), (fig3, ax3)]

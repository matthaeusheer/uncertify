from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch

from uncertify.data.dataloaders import DataLoader
from uncertify.visualization.plotting import setup_plt_figure
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Iterable, Generator, Dict


def plot_pixel_histogram(arrays: List[np.ndarray],
                         labels: List[str] = None, **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:
    fig, ax = setup_plt_figure(aspect='auto', **kwargs)
    hist_kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=20)
    hist_kwargs.update({key: value for key, value in kwargs.items() if key in hist_kwargs})
    return_tuples = []
    for idx, array in enumerate(arrays):
        return_tuple = ax.hist(array, label=labels[idx] if labels is not None else None, **hist_kwargs)
        return_tuples.append(return_tuple)
    if labels is not None:
        ax.legend()
    return return_tuples


def plot_loss_histograms(output_generators: Iterable[Generator[Dict[str, Tensor], None, None]],
                         names: Iterable[str], **kwargs) -> Tuple[plt.Figure, plt.Axes]:
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
        kld_arrays.append(kld_losses[name])
        kld_labels.append(f' '.join([name, 'KLD']))
        rec_arrays.append(rec_losses[name])
        rec_labels.append(f' '.join([name, 'REC']))
    plot_pixel_histogram(kld_arrays, kld_labels, xlabel='KLD Term Loss', **kwargs)
    plot_pixel_histogram(rec_arrays, rec_labels, xlabel='Reconstruction Loss', **kwargs)

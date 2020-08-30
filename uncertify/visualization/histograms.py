import numpy as np
import matplotlib.pyplot as plt

from uncertify.visualization.plotting import setup_plt_figure

from typing import Tuple, List


def plot_pixel_histogram(arrays: List[np.ndarray], **kwargs) -> List[Tuple[plt.Figure, plt.Axes]]:
    fig, ax = setup_plt_figure(aspect='auto', **kwargs)
    hist_kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=10)
    hist_kwargs.update({key: value for key, value in kwargs.items() if key in hist_kwargs})
    return_tuples = []
    for array in arrays:
        return_tuple = ax.hist(array, **hist_kwargs)
        return_tuples.append(return_tuple)
    return return_tuples

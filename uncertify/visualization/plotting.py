"""
Some functionality to quickly set up matplotlib figures and plot images (numpy.ndarray).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def setup_plt_figure(**kwargs) -> (plt.Figure, plt.Axes):
    """Create a default matplotlib figure and return the figure and ax object.
    Possible kwargs which are handled by this function (matplotlib uses kwargs internally so there is not really
    a way around this):
        figsize: Tuple(float, float), width and height in inches, defaults to (6.4, 4.8)
        title: str, sets the center title for the figures axes
    """
    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs.get('figsize'))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title'), fontweight='bold')
    if 'aspect' in kwargs:
        ax.set_aspect(kwargs.get('aspect'))
    else:
        ax.set_aspect('auto')
    if 'axis' in kwargs:
        ax.axis(kwargs.get('axis'))
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs.get('xlabel'), fontweight='bold')
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs.get('ylabel'), fontweight='bold')
    return fig, ax


def imshow(img: np.ndarray, axis: str = 'on', ax: plt.Axes = None, add_colorbar: bool = True,
           **kwargs) -> (plt.Figure, plt.Axes):
    """Imshow wrapper to plot images in various image spaces. Constructs a figure object under the hood.
    Arguments
    ---------
        img: the input image
        axis: whether to display the axis with ticks and everything by default
        ax: if None, create new plt Axes, otherwise take this one for plotting
        kwargs: those will be forwarded into the setup_plt_figure function
    Returns
    -------
        fig, ax: a tuple of a matplotlib Figure and Axes object
    """

    if ax is None:
        fig, ax = setup_plt_figure(**kwargs)
    else:
        fig = ax.get_figure()
        ax.set_title(kwargs.get('title', ''))
    ax.axis(axis)
    imshow_kwargs = {'cmap': kwargs['cmap'] if 'cmap' in kwargs else 'hot'}
    for key in ['vmin', 'vmax']:
        if key in kwargs:
            imshow_kwargs[key] = kwargs.get(key)
    im = ax.imshow(img, **imshow_kwargs)
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 12}
    matplotlib.rc('font', **font)
    return fig, ax

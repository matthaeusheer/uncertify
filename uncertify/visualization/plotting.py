"""
Some functionality to quickly set up matplotlib figures and plot images (numpy.ndarray).
"""
import numpy as np
import matplotlib.pyplot as plt


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
        ax.set_title(kwargs.get('title'))
    ax.set_aspect('equal')
    return fig, ax


def imshow(img: np.ndarray, axis: str = 'on', ax: plt.Axes = None, **kwargs) -> (plt.Figure, plt.Axes):
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

    cmap = kwargs['cmap'] if 'cmap' in kwargs else 'hot'
    imshow_kwargs = {}
    for key in ['vmin', 'vmax']:
        if key in kwargs:
            imshow_kwargs[key] = kwargs.get(key)
    ax.imshow(img, cmap=cmap, vmin=100)
    return fig, ax

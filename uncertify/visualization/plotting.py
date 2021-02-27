import logging
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

LOG = logging.getLogger(__name__)

DEFAULT_SAVE_FIG_KWARGS = dict(dpi=600, transparent=True, bbox_inches='tight')


def set_matplotlib_rc() -> None:
    """Basically set font settings etc. for nice looking figures."""
    matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix', })
    matplotlib.rcParams.update({'font.size': 22, 'legend.fontsize': 15})
    font = {'weight': 'bold'}
    matplotlib.rc('font', **font)

    plt.rcParams['axes.linewidth'] = 2


def setup_plt_figure(**kwargs) -> (plt.Figure, plt.Axes):
    """Create a default matplotlib figure and return the figure and ax object.
    Possible kwargs which are handled by this function (matplotlib uses kwargs internally so there is not really
    a way around this):
        figsize: Tuple(float, float), width and height in inches, defaults to (6.4, 4.8)
        title: str, sets the center title for the figures axes
    """
    set_matplotlib_rc()
    if 'figsize' in kwargs:
        fig = plt.figure(figsize=kwargs.get('figsize'))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title'), pad=15)
    if 'aspect' in kwargs:
        ax.set_aspect(kwargs.get('aspect'))
    else:
        ax.set_aspect('auto')
    if 'axis' in kwargs:
        ax.axis(kwargs.get('axis'))
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs.get('xlabel'))
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs.get('ylabel'))
    if 'xmin' in kwargs:
        ax.set_xlim(left=kwargs.get('xmin'))
    if 'ymin' in kwargs:
        ax.set_ylim(left=kwargs.get('ymin'))
    if 'xmax' in kwargs:
        ax.set_xlim(right=kwargs.get('xmax'))
    if 'ymax' in kwargs:
        ax.set_ylim(right=kwargs.get('ymax'))

    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')

    plt.tight_layout()
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


def save_fig(fig: plt.Figure, save_path: Path, create_dirs: bool = False, **kwargs) -> None:
    """Save a figure and potentially create the location. """
    if not save_path.parent.exists():
        if create_dirs:
            save_path.parent.mkdir(parents=True)
        else:
            LOG.warning(f'Save storage location does not exist. Consider create_dirs=True. Not saving figure.')
            return
    DEFAULT_SAVE_FIG_KWARGS.update(kwargs)
    fig.savefig(save_path, **DEFAULT_SAVE_FIG_KWARGS)
    LOG.debug(f'Saved figure at {save_path} with settings: {DEFAULT_SAVE_FIG_KWARGS}')

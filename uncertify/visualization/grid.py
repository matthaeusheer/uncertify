import numpy as np
import matplotlib.pyplot as plt

from uncertify.visualization.plotting import imshow, setup_plt_figure
from uncertify.utils.custom_types import Tensor

from typing import Tuple


def imshow_grid(grid_tensor: Tensor, one_channel: bool = False, plt_show: bool = True, **plt_kwargs) -> Tuple[plt.Figure,
                                                                                                              plt.Axes]:
    """Does an imshow on an grid returned by torchvision.utils.make_grid()."""
    fig, ax = setup_plt_figure(**plt_kwargs)
    if one_channel:
        grid_tensor = grid_tensor.mean(dim=0)
    np_img = grid_tensor.numpy()
    if one_channel:
        imshow(np_img, ax=ax, **plt_kwargs)
    else:
        imshow(np.transpose(np_img, (1, 2, 0)), ax=ax, **plt_kwargs)
    if plt_show:
        plt.show()
    return fig, ax


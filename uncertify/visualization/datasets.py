from itertools import islice
import logging

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from uncertify.visualization.grid import imshow_grid

LOG = logging.getLogger(__name__)


def plot_brats_batches(brats_dataloader: DataLoader, plot_n_batches: int) -> None:
    LOG.info('Plotting BraTS2017 Dataset')
    for sample in islice(brats_dataloader, plot_n_batches):
        grid = make_grid(
            torch.cat((sample['scan'].type(torch.FloatTensor), sample['seg'].type(torch.FloatTensor)), dim=2))
        imshow_grid(grid, one_channel=True, plt_show=True, cmap='hot', figsize=(9, 8), axis='off')


def plot_camcan_batches(camcan_dataloader: DataLoader, plot_n_batches: int) -> None:
    LOG.info('Plotting CamCAN Dataset')
    for sample in islice(camcan_dataloader, plot_n_batches):
        grid = make_grid(sample['scan'].type(torch.FloatTensor))
        imshow_grid(grid, one_channel=True, plt_show=True, cmap='hot', figsize=(9, 8), axis='off')

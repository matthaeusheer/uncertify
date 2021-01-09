import logging

import torch
import torchvision
import numpy as np

from uncertify.utils.python_helpers import print_dict_tree
from uncertify.utils.python_helpers import get_idx_of_closest_value
from uncertify.visualization.grid import imshow_grid

LOG = logging.getLogger(__name__)


def plot_entropy_samples_over_range(metrics_ood_dict: dict, dataset_name: str,
                                    start_val: float, end_val: float, n_values: int, **plt_kwargs) -> None:
    """Given a metrics_ood_dict, whose keys have to be in the format score -> dataset -> ..., plot the entropy
    plot a grid of images starting from top left to bottom right in rows left to right with increasing dose kde
    entropy.
    """
    try:
        ood_dict = metrics_ood_dict['dose'][dataset_name]
        healthy_entropy = ood_dict['dose_kde_healthy']['entropy']
        lesional_entropy = ood_dict['dose_kde_lesional']['entropy']
        healthy_scans = ood_dict['healthy_scans']
        lesional_scans = ood_dict['lesional_scans']
    except KeyError as err:
        print(f'The metrics_ood_dict does not have the correct keys to generate your plot.'
              f'Given:\n{print_dict_tree(metrics_ood_dict)}')
        raise err

    LOG.info(f'Plotting scans with entropy from {start_val:.1f}-{end_val:.1f}.')
    LOG.info(f'Min/max entropy from all healthy scans is: {min(healthy_entropy):.1f}/{max(healthy_entropy):.1f}')
    LOG.info(f'Min/max entropy from all lesional scans is: {min(lesional_entropy):.1f}/{max(lesional_entropy):.1f}')

    ref_values = np.linspace(start_val, end_val, n_values)
    healthy_img_batch = torch.zeros(size=[n_values, 1, 128, 128])
    lesional_img_batch = torch.zeros(size=[n_values, 1, 128, 128])

    for img_idx, ref_val in enumerate(ref_values):
        closest_healthy_idx = get_idx_of_closest_value(healthy_entropy, ref_val)
        closest_lesional_idx = get_idx_of_closest_value(lesional_entropy, ref_val)

        lesional_img = lesional_scans[closest_lesional_idx]
        healthy_img = healthy_scans[closest_healthy_idx]

        lesional_img_batch[img_idx] = lesional_img
        healthy_img_batch[img_idx] = healthy_img

    healthy_grid = torchvision.utils.make_grid(healthy_img_batch)
    lesional_grid = torchvision.utils.make_grid(lesional_img_batch)
    imshow_grid(healthy_grid, one_channel=True,
                title=f'Healthy entropy [{start_val:.1f}-{end_val:.1f}]', **plt_kwargs)
    imshow_grid(lesional_grid, one_channel=True,
                title=f'Lesional entropy [{start_val:.1f}-{end_val:.1f}]', **plt_kwargs)



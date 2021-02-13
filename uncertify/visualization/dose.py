import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torchvision

from uncertify.common import DATA_DIR_PATH

from typing import List

from uncertify.utils.python_helpers import print_dict_tree, get_idx_of_closest_value
from uncertify.visualization.grid import imshow_grid

LABEL_MAP = {
    'rec_err': '$\ell_{1}$',
    'kl_div': '$D_{KL}$',
    'elbo': '$\mathcal{L}$',
    'entropy': '$H_{\ell_{1}}$'
}

LOG = logging.getLogger(__name__)


def do_pair_plot_statistics(statistics_dict: dict, dose_statistics: List[str],
                            dataset_name: str, hue: str = 'is_lesional') -> sns.PairGrid:
    """
    Arguments
    ---------
        statistics_dict: dictionary as returned by aggregate_slice_wise_statistics
        dose_statistics: statistics to use in the plot
        dataset_name: name of the dataset used for file naming
        hue: which column in the dataframe to use as hue
    """
    stat_df = pd.DataFrame(statistics_dict)
    grid = sns.pairplot(stat_df, vars=dose_statistics, corner=True, plot_kws={"s": 10}, palette='viridis',
                        hue=hue, diag_kws={'shade': False}, diag_kind='kde')
    grid.map_lower(sns.kdeplot, shade=True, thresh=0.05, alpha=0.7)

    if hue is not None:
        grid._legend.set_title('')
        new_labels = ['healthy', 'lesional']
        for t, l in zip(grid._legend.texts, new_labels):
            t.set_text(l)
    # Set nice x and y labels
    for ax in grid.axes.flatten():
        if ax is not None:
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            old_xlabel = ax.get_xlabel()
            old_ylabel = ax.get_ylabel()
            if old_xlabel in LABEL_MAP:
                ax.set_xlabel(LABEL_MAP[old_xlabel])
            if old_ylabel in LABEL_MAP:
                ax.set_ylabel(LABEL_MAP[old_ylabel])
    grid.tight_layout()
    grid.savefig(DATA_DIR_PATH / 'plots' / f'dose_pairplot_{dataset_name}.png')
    return grid


def plot_dose_samples_over_range(metrics_ood_dict: dict, dataset_name: str, mode: str, stat_type: str,
                                 start_val: float, end_val: float, n_values: int, **plt_kwargs) -> None:
    """Given a metrics_ood_dict, whose keys have to be in the format score -> dataset -> ..., plot the statistic value
    plot a grid of images starting from top left to bottom right in rows left to right with increasing dose kde
    value.
    """
    if mode not in ['dose_kde', 'dose_stat']:
        raise ValueError(f'Chose the mode, such that it is either "dose_kde" or "raw" (statistic value).')
    try:
        ood_dict = metrics_ood_dict['dose'][dataset_name]
        healthy_values = ood_dict[f'{mode}_healthy'][stat_type]
        lesional_values = ood_dict[f'{mode}_lesional'][stat_type]
        healthy_scans = ood_dict['healthy_scans']
        lesional_scans = ood_dict['lesional_scans']
    except KeyError as err:
        print(f'The metrics_ood_dict does not have the correct keys to generate your plot.'
              f'Given:\n{print_dict_tree(metrics_ood_dict)}')
        raise err

    LOG.info(f'Plotting scans with {stat_type} from {start_val:.1f}-{end_val:.1f}.')
    LOG.info(
        f'Min/max {mode} {stat_type} from all healthy scans is: {min(healthy_values):.1f}/{max(healthy_values):.1f}')
    LOG.info(
        f'Min/max {mode} {stat_type} from all lesional scans is: {min(lesional_values):.1f}/{max(lesional_values):.1f}')
    ref_values = np.linspace(start_val, end_val, n_values)
    healthy_img_batch = torch.zeros(size=[n_values, 1, 128, 128])
    lesional_img_batch = torch.zeros(size=[n_values, 1, 128, 128])

    for img_idx, ref_val in enumerate(ref_values):
        closest_healthy_idx = get_idx_of_closest_value(healthy_values, value=ref_val)
        closest_lesional_idx = get_idx_of_closest_value(lesional_values, value=ref_val)

        lesional_img = lesional_scans[closest_lesional_idx]
        healthy_img = healthy_scans[closest_healthy_idx]

        lesional_img_batch[img_idx] = lesional_img
        healthy_img_batch[img_idx] = healthy_img

    healthy_grid = torchvision.utils.make_grid(healthy_img_batch)
    lesional_grid = torchvision.utils.make_grid(lesional_img_batch)
    imshow_grid(healthy_grid, title=f'Healthy {mode} {stat_type} [{start_val:.1f}-{end_val:.1f}]', **plt_kwargs)
    imshow_grid(lesional_grid, title=f'Lesional {mode} {stat_type} [{start_val:.1f}-{end_val:.1f}]', **plt_kwargs)

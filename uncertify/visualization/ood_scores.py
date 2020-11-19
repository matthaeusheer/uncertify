import logging

import torchvision
import numpy as np

from uncertify.visualization.histograms import plot_multi_histogram
from uncertify.utils.python_helpers import get_indices_of_n_largest_items, get_indices_of_n_smallest_items
from uncertify.utils.python_helpers import get_idx_of_closest_value
from uncertify.visualization.grid import imshow_grid
from uncertify.common import DATA_DIR_PATH

LOG = logging.getLogger(__name__)


def plot_ood_scores(ood_dataset_dict: dict, score_label: str = 'WAIC', dataset_name_filters: list = None,
                    modes_to_include: list = None, do_save: bool = True) -> None:
    """Plot OOD score distribution histogram for different datasets, all, healthy and/or unhealthy.

    Arguments
        ood_dataset_dict: a dictionary with dataset names as keys and a dict like
                         {'all': [scores], 'healthy': [scores], ...}
        score_label: the name of the OOD score used
        dataset_name_filters: a list of words for which datasets are excluded if some are in their name
        modes_to_include: a list with 'all', 'healthy', 'lesional' potential entries, if None, all will be considered
    """
    if dataset_name_filters is None:
        dataset_name_filters = []
    if modes_to_include is None:
        modes_to_include = ['all', 'lesional', 'healthy']

    waic_lists = []
    list_labels = []

    for dataset_name, sub_dict in ood_dataset_dict.items():
        if any([filter_word in dataset_name for filter_word in dataset_name_filters]):
            continue
        has_only_healthy = len(sub_dict['lesional']) == 0
        if has_only_healthy:
            waic_scores = sub_dict['healthy']
            label = f'{dataset_name}'
            list_labels.append(label)
            waic_lists.append(waic_scores)
        else:
            for mode in modes_to_include:
                waic_scores = sub_dict[mode]
                label = f'{dataset_name} {mode}'
                list_labels.append(label)
                waic_lists.append(waic_scores)

    fig, _ = plot_multi_histogram(waic_lists, list_labels, plot_density=False,
                                  figsize=(12, 6), xlabel=score_label, ylabel='Slice-wise frequency',
                                  hist_kwargs={'bins': 17})
    if do_save:
        save_path = DATA_DIR_PATH / 'plots' / 'waic_scores.png'
        fig.savefig(save_path)
        LOG.info(f'Saved OOD score figure at: {save_path}')


def plot_most_least_ood(waic_dict: dict, dataset_name: str, n_most: int = 16, do_lesional: bool = True) -> None:
    """For healthy and lesional samples, plot the ones which are most and least OOD."""
    ood_dict = waic_dict[dataset_name]

    def create_ood_grids(healthy_leasional: str):
        scores = ood_dict[healthy_leasional]
        slices = ood_dict[f'{healthy_leasional}_scans']
        largest_score_indices = get_indices_of_n_largest_items(scores, n_most)
        smallest_score_indices = get_indices_of_n_smallest_items(scores, n_most)

        largest_slices = [slices[idx] for idx in largest_score_indices]
        smallest_slices = [slices[idx] for idx in smallest_score_indices]

        largest_grid = torchvision.utils.make_grid(largest_slices, padding=0, normalize=False)
        smallest_grid = torchvision.utils.make_grid(smallest_slices, padding=0, normalize=False)

        return largest_grid, smallest_grid

    LOG.debug('Creating healthy grids...')
    most_ood_healthy_grid, least_ood_healthy_grid = create_ood_grids('healthy')
    if do_lesional:
        LOG.debug('Creating lesional grids...')
        most_ood_lesional_grid, least_ood_lesional_grid = create_ood_grids('lesional')

    imshow_grid(most_ood_healthy_grid, one_channel=True, figsize=(12, 8), title=f'Most OOD Healthy {dataset_name}',
                axis='off')
    imshow_grid(least_ood_healthy_grid, one_channel=True, figsize=(12, 8), title=f'Least OOD Healthy {dataset_name}',
                axis='off')
    if do_lesional:
        imshow_grid(most_ood_lesional_grid, one_channel=True, figsize=(12, 8),
                    title=f'Most OOD Lesional {dataset_name}', axis='off')
        imshow_grid(least_ood_lesional_grid, one_channel=True, figsize=(12, 8),
                    title=f'Least OOD Lesional {dataset_name}', axis='off')


def plot_samples_close_to_score(ood_dict: dict, dataset_name: str, min_score: float, max_score: float, n: int = 32,
                                do_lesional: bool = True) -> None:
    """Arrange slices in a grid such that each slice displayed is closest to the interpolation OOD score from
    linspace which goes from min_score to max_score with n samples."""
    ood_dict = ood_dict[dataset_name]
    ref_scores = np.linspace(min_score, max_score, n)

    def create_ood_grids(healthy_leasional: str):
        scores = ood_dict[healthy_leasional]
        slices = ood_dict[f'{healthy_leasional}_scans']

        final_scores = []
        final_slices = []

        for ref_score in ref_scores:
            scores_idx = get_idx_of_closest_value(scores, ref_score)
            final_scores.append(scores[scores_idx])
            final_slices.append(slices[scores_idx])

        return torchvision.utils.make_grid(final_slices, padding=0, normalize=False)

    healthy_grid = create_ood_grids('healthy')
    if do_lesional:
        lesional_grid = create_ood_grids('lesional')

    imshow_grid(healthy_grid, one_channel=True, figsize=(12, 8),
                title=f'Healthy {dataset_name} {min_score}-{max_score}', axis='off')
    if do_lesional:
        imshow_grid(lesional_grid, one_channel=True, figsize=(12, 8),
                    title=f'Lesional {dataset_name} {min_score}-{max_score}', axis='off')
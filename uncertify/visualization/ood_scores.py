import logging

import torch
import torchvision
import numpy as np

from uncertify.visualization.dose import LOG
from uncertify.visualization.histograms import plot_multi_histogram
from uncertify.utils.python_helpers import get_indices_of_n_largest_items, get_indices_of_n_smallest_items, \
    print_dict_tree, get_idx_of_closest_value
from uncertify.utils.python_helpers import get_idx_of_closest_value
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.evaluation.utils import mask_background_to_zero
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
            ood_scores = sub_dict['healthy']
            label = f'{dataset_name}'
            list_labels.append(label)
            waic_lists.append(ood_scores)
        else:
            for mode in modes_to_include:
                ood_scores = sub_dict[mode]
                label = f'{dataset_name} {mode}'
                list_labels.append(label)
                waic_lists.append(ood_scores)

    fig, _ = plot_multi_histogram(waic_lists, list_labels, plot_density=False,
                                  figsize=(12, 6), xlabel=score_label, ylabel='Slice-wise frequency',
                                  hist_kwargs={'bins': 17})
    if do_save:
        save_path = DATA_DIR_PATH / 'plots' / f'{score_label}.png'
        fig.savefig(save_path)
        LOG.info(f'Saved OOD score figure at: {save_path}')


def plot_most_least_ood(ood_dict: dict, dataset_name: str, n_most: int = 16, do_lesional: bool = True,
                        small_score_is_more_odd: bool = True) -> None:
    """For healthy and lesional samples, plot the ones which are most and least OOD."""
    ood_dict = ood_dict[dataset_name]

    def create_ood_grids(healthy_leasional: str):
        scores = ood_dict[healthy_leasional]
        slices = ood_dict[f'{healthy_leasional}_scans']
        masks = ood_dict['masks']

        slices = [normalize_to_0_1(s) for s in slices]
        slices = [mask_background_to_zero(s, m) for s, m in zip(slices, masks)]

        largest_score_indices = get_indices_of_n_largest_items(scores, n_most)
        smallest_score_indices = get_indices_of_n_smallest_items(scores, n_most)
        largest_scores = [scores[idx] for idx in largest_score_indices]
        smallest_scores = [scores[idx] for idx in smallest_score_indices]

        largest_slices = [slices[idx] for idx in largest_score_indices]
        smallest_slices = [slices[idx] for idx in smallest_score_indices]

        largest_values_grid = normalize_to_0_1(torchvision.utils.make_grid(largest_slices, padding=0, normalize=False))
        smallest_values_grid = normalize_to_0_1(torchvision.utils.make_grid(smallest_slices, padding=0, normalize=False))
        if small_score_is_more_odd:
            return smallest_values_grid, largest_values_grid, smallest_scores, largest_scores
        else:
            return largest_values_grid, smallest_values_grid, largest_scores, smallest_scores

    LOG.debug('Creating healthy grids...')
    most_ood_healthy_grid, least_ood_healthy_grid, most_ood_score_healthy, least_ood_scores_healthy = create_ood_grids('healthy')
    if do_lesional:
        LOG.debug('Creating lesional grids...')
        most_ood_lesional_grid, least_ood_lesional_grid, most_ood_score_lesional, least_ood_scores_lesional = create_ood_grids('lesional')

    print(most_ood_score_healthy)
    imshow_grid(most_ood_healthy_grid, one_channel=True, figsize=(12, 8), title=f'Most OOD Healthy {dataset_name}',
                axis='off')
    print(least_ood_scores_healthy)
    imshow_grid(least_ood_healthy_grid, one_channel=True, figsize=(12, 8), title=f'Least OOD Healthy {dataset_name}',
                axis='off')
    if do_lesional:
        print(most_ood_score_lesional)
        imshow_grid(most_ood_lesional_grid, one_channel=True, figsize=(12, 8),
                    title=f'Most OOD Lesional {dataset_name}', axis='off')
        print(least_ood_scores_lesional)
        imshow_grid(least_ood_lesional_grid, one_channel=True, figsize=(12, 8),
                    title=f'Least OOD Lesional {dataset_name}', axis='off')


def plot_samples_close_to_score(ood_dict: dict, dataset_name: str, min_score: float, max_score: float, n: int = 32,
                                do_lesional: bool = True, show_ground_truth: bool = False,
                                print_score: bool = False) -> None:
    """Arrange slices in a grid such that each slice displayed is closest to the interpolation OOD score from
    linspace which goes from min_score to max_score with n samples."""
    ood_dict = ood_dict[dataset_name]
    ref_scores = np.linspace(min_score, max_score, n)

    def create_ood_grids(healthy_leasional: str):
        scores = ood_dict[healthy_leasional]
        slices = ood_dict[f'{healthy_leasional}_scans']
        masks = ood_dict['masks']
        segmentations = ood_dict[f'{healthy_leasional}_segmentations']
        final_scores = []
        final_slices = []
        final_masks = []
        final_segmentations = []

        for ref_score in ref_scores:
            scores_idx = get_idx_of_closest_value(scores, ref_score)
            final_scores.append(scores[scores_idx])
            final_slices.append(slices[scores_idx])
            final_masks.append(masks[scores_idx])
            if show_ground_truth:
                final_segmentations.append(segmentations[scores_idx])

        final_slices = [normalize_to_0_1(s) for s in final_slices]
        final_slices = [mask_background_to_zero(s, m) for s, m in zip(final_slices, final_masks)]

        slices_grid = torchvision.utils.make_grid(final_slices, padding=0, normalize=False)
        segmentations_grid = None
        if show_ground_truth:
            segmentations_grid = torchvision.utils.make_grid(final_segmentations, padding=0, normalize=False)
        if print_score:
            formatted_scores = [f'{val:.2f}' for val in final_scores]
            LOG.info(f'Scores: {formatted_scores}')
        return slices_grid, segmentations_grid

    healthy_slices_grid, healthy_segmentations_grid = create_ood_grids('healthy')
    imshow_grid(healthy_slices_grid, one_channel=True, figsize=(12, 8),
                title=f'Healthy {dataset_name} {min_score}-{max_score}', axis='off')
    if show_ground_truth:
        imshow_grid(healthy_segmentations_grid, one_channel=True, figsize=(12, 8),
                    title=f'Healthy Ground Truth {dataset_name} {min_score}-{max_score}', axis='off')
    if do_lesional:
        lesional_slices_grid, lesional_segmentations_grid = create_ood_grids('lesional')
        imshow_grid(lesional_slices_grid, one_channel=True, figsize=(12, 8),
                    title=f'Lesional {dataset_name} {min_score}-{max_score}', axis='off')
        if show_ground_truth:
            imshow_grid(lesional_segmentations_grid, one_channel=True, figsize=(12, 8),
                        title=f'Lesional Ground Truth {dataset_name} {min_score}-{max_score}', axis='off')


def plot_ood_samples_over_range(metrics_ood_dict: dict, dataset_name: str, mode: str, stat_type: str,
                                start_val: float, end_val: float, n_values: int, **plt_kwargs) -> None:
    """Given a metrics_ood_dict, whose keys have to be in the format score -> dataset -> ..., plot the statistic value
    plot a grid of images starting from top left to bottom right in rows left to right with increasing dose kde
    value.
    """
    if mode not in ['dose_kde', 'dose_stat', 'waic']:
        raise ValueError(f'Chose the mode, such that it is either "dose_kde" or "raw" (statistic value).')
    if 'dose' in mode:
        main_mode = 'dose'
    else:
        main_mode = 'waic'
    try:
        ood_dict = metrics_ood_dict[main_mode][dataset_name]
        if main_mode == 'dose':
            healthy_values = ood_dict[f'{mode}_healthy'][stat_type]
            lesional_values = ood_dict[f'{mode}_lesional'][stat_type]
        elif main_mode == 'waic':
            healthy_values = ood_dict[f'healthy']
            lesional_values = ood_dict[f'lesional']
        else:
            raise KeyError(f'main_mode not supported')
        healthy_scans = ood_dict['healthy_scans']
        lesional_scans = ood_dict['lesional_scans']
        healthy_recs = ood_dict['healthy_reconstructions']
        lesional_recs = ood_dict['lesional_reconstructions']
    except KeyError as err:
        print(f'The metrics_ood_dict does not have the correct keys to generate your plot.'
              f'Given:\n{print_dict_tree(metrics_ood_dict)}')
        raise err

    mode_description = f'{main_mode} {stat_type if main_mode=="dose" else ""}'
    LOG.info(f'Plotting scans with {stat_type} from {start_val:.1f}-{end_val:.1f}.')
    LOG.info(
        f'Min/max {mode_description} from all healthy scans is: {min(healthy_values):.1f}/{max(healthy_values):.1f}')
    LOG.info(
        f'Min/max {mode_description} from all lesional scans is: {min(lesional_values):.1f}/{max(lesional_values):.1f}')
    # Will fill up two tensors with healthy and lesional images
    healthy_img_batch = torch.zeros(size=[n_values, 1, 128, 128])
    lesional_img_batch = torch.zeros(size=[n_values, 1, 128, 128])
    healthy_img_rec_batch = torch.zeros(size=[n_values, 1, 128, 128])
    lesional_img_rec_batch = torch.zeros(size=[n_values, 1, 128, 128])

    # Define reference values and initialize actual values of picked images
    ref_values = np.linspace(start_val, end_val, n_values)
    picked_healthy_values = []
    picked_lesional_values = []

    for img_idx, ref_val in enumerate(ref_values):
        closest_healthy_idx = get_idx_of_closest_value(healthy_values, value=ref_val)
        closest_lesional_idx = get_idx_of_closest_value(lesional_values, value=ref_val)

        lesional_img = lesional_scans[closest_lesional_idx]
        healthy_img = healthy_scans[closest_healthy_idx]
        lesional_rec_img = lesional_recs[closest_lesional_idx]
        healthy_rec_img = healthy_recs[closest_healthy_idx]

        lesional_img_batch[img_idx] = lesional_img
        healthy_img_batch[img_idx] = healthy_img
        lesional_img_rec_batch[img_idx] = lesional_rec_img
        healthy_img_rec_batch[img_idx] = healthy_rec_img
        picked_healthy_values.append(healthy_values[closest_healthy_idx])
        picked_lesional_values.append(lesional_values[closest_lesional_idx])

    nrow = plt_kwargs.get('nrow', 8)
    healthy_grid = torchvision.utils.make_grid(healthy_img_batch, nrow=nrow)
    healthy_rec_grid = torchvision.utils.make_grid(healthy_img_rec_batch, nrow=nrow)

    lesional_grid = torchvision.utils.make_grid(lesional_img_batch, nrow=nrow)
    lesional_rec_grid = torchvision.utils.make_grid(lesional_img_rec_batch, nrow=nrow)

    imshow_grid(healthy_grid, title=f'Healthy {mode_description} [{start_val:.1f}, {end_val:.1f}]', **plt_kwargs)
    imshow_grid(healthy_rec_grid, title=f'Healthy {mode_description} [{start_val:.1f}, {end_val:.1f}]', **plt_kwargs)
    imshow_grid(lesional_grid, title=f'Lesional {mode_description} [{start_val:.1f}, {end_val:.1f}]', **plt_kwargs)
    imshow_grid(lesional_rec_grid, title=f'Lesional {mode_description} [{start_val:.1f}, {end_val:.1f}]', **plt_kwargs)
    LOG.info(f'Healthy values:\n{picked_healthy_values}')
    LOG.info(f'Healthy values:\n{picked_lesional_values}')
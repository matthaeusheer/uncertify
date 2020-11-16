import logging

from uncertify.visualization.histograms import plot_multi_histogram
from uncertify.common import DATA_DIR_PATH

LOG = logging.getLogger(__name__)


def plot_ood_scores(ood_dataset_dict: dict, score_label: str = 'WAIC', dataset_name_filters: list = None,
                    modes_to_include: list = None, do_save: bool = True) -> None:
    """Plot OOD score distribution for different datasets, all, healthy and/or unhealthy.

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

    print(list_labels)
    fig, _ = plot_multi_histogram(waic_lists, list_labels, plot_density=False,
                                  figsize=(12, 6), xlabel=score_label, ylabel='Slice-wise frequency',
                                  hist_kwargs={'bins': 17})
    if do_save:
        save_path = DATA_DIR_PATH / 'plots' / 'waic_scores.png'
        fig.savefig(save_path)
        LOG.info(f'Saved OOD score figure at: {save_path}')

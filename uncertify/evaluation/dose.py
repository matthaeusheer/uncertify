from collections import defaultdict

from scipy.stats import gaussian_kde
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.statistics import aggregate_slice_wise_statistics
from uncertify.evaluation.statistics import STATISTICS_FUNCTIONS

from typing import List, Dict


def compute_slice_wise_dose_kde_scores(model: nn.Module, test_dataloader: DataLoader,
                                       kde_func_dict: Dict[str, gaussian_kde], statistics: List[str],
                                       max_n_batches: int = None) -> Dict[str, List[float]]:
    """Computes the per-slice DoSE KDE scores which are simply the KDE fit values for inferred samples."""
    for stat_name in statistics:
        if stat_name not in kde_func_dict.keys():
            raise ValueError(f'Attempting to test for statistic for which we have no KDE fit. Fitted statistics '
                             f'are {list(kde_func_dict.keys())}')
    # First, aggregate statistic values for test samples which we then run through the KDE fit to get the DOSE_KDE score
    test_stat_dict = aggregate_slice_wise_statistics(model, test_dataloader, statistics, max_n_batches)

    # Run inferred sample-wise statistic values through KDE fit
    dose_kde_dict = defaultdict(list)
    filtered_test_stat_dict = {key: values for key, values in test_stat_dict.items()
                               if key in STATISTICS_FUNCTIONS.keys()}
    for stat_name, test_stat_values in filtered_test_stat_dict.items():
        kde_values = kde_func_dict[stat_name](test_stat_values)
        dose_kde_dict[stat_name] = kde_values
    if 'is_lesional' in test_stat_dict.keys():
        dose_kde_dict.update({'is_lesional': test_stat_dict['is_lesional']})
    return dose_kde_dict


def compute_slice_wise_dose_scores(dose_kde_dict: Dict[str, List[float]]) -> List[float]:
    """Construct KDE score by summing up the different KDE fit values for a sample."""
    slice_wise_dose_scores = []
    for dose_kde_values in zip(*dose_kde_dict.values()):
        slice_wise_dose_scores.append(sum(dose_kde_values))
    return slice_wise_dose_scores

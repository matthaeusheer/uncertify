from collections import defaultdict

from scipy.stats import gaussian_kde
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.statistics import aggregate_slice_wise_statistics, fit_statistics
from uncertify.evaluation.statistics import STATISTICS_FUNCTIONS

from typing import List, Dict, Tuple, Optional


def compute_slice_wise_dose_kde_scores(model: nn.Module, test_dataloader: DataLoader,
                                       kde_func_dict: Dict[str, gaussian_kde], statistics: List[str],
                                       max_n_batches: int = None) -> Dict[str, Optional[List[float]]]:
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
    dose_kde_dict.update({'is_lesional': test_stat_dict['is_lesional']})
    return dose_kde_dict


def compute_slice_wise_dose_scores(dose_kde_dict: Dict[str, List[float]]) -> List[float]:
    """Construct KDE score by summing up the different KDE fit values for a sample."""
    slice_wise_dose_scores = []
    for dose_kde_values in zip(*{key: values for key, values in dose_kde_dict.items() if values is not None}.values()):
        slice_wise_dose_scores.append(sum(dose_kde_values))
    return slice_wise_dose_scores


def full_pipeline_slice_wise_dose_scores(train_dataloader: DataLoader, test_dataloader: DataLoader,
                                         model: nn.Module, statistics: List[str], max_n_batches: int = None,
                                         kde_func_dict: Dict[str, gaussian_kde] = None) -> Tuple[List[float],
                                                                                                 Dict[str, List[float]]]:
    """Run the whole DoSE pipeline to get slice-wise dose scores for samples from test_dataloader.

    Arguments
        train_dataloader: the data which has been used to train the model and to fit the KDE on statistics
        test_dataloader: the data for which to obtain dose scores
        mode: trained model
        statistics: list of strings of statistics keys to run dose for
        max_n_batches: trivial
        kde_func_dict: if not None, will use this one instead of fitting new KDEs on training data
    Returns
        dose_scores: a list of float values holding the final DoSE score for each slice
        dose_kde_dict: a dictionary holding the dose kde scores for each slice for each statistic
    """
    if kde_func_dict is None:
        statistics_dict = aggregate_slice_wise_statistics(model, train_dataloader, statistics, max_n_batches)
        kde_func_dict = fit_statistics(statistics_dict)
    dose_kde_dict = compute_slice_wise_dose_kde_scores(model, test_dataloader, kde_func_dict, statistics, max_n_batches)
    dose_scores = compute_slice_wise_dose_scores(dose_kde_dict)
    return dose_scores, dose_kde_dict


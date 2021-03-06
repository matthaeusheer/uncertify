from collections import defaultdict

from scipy.stats import gaussian_kde
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.statistics import aggregate_slice_wise_statistics, fit_statistics
from uncertify.evaluation.statistics import STATISTICS_FUNCTIONS

from typing import List, Dict, Tuple, Optional, Any


def compute_slice_wise_dose_kde_scores(model: nn.Module, test_dataloader: DataLoader,
                                       kde_func_dict: Dict[str, gaussian_kde], statistics: Tuple[str],
                                       max_n_batches: int = None) -> Tuple[Dict[str, Optional[List[float]]],
                                                                           Dict[str, Any]]:
    """Computes the per-slice DoSE KDE scores which are simply the KDE fit values for inferred samples.
    Note: Additionally to Dose_KDE scores, also lesional bools and scans/masks/segmentations are passed along.
    """
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
        kde_values = kde_func_dict[stat_name](test_stat_values)  # calculate values under fit
        dose_kde_dict[stat_name] = kde_values

    return dose_kde_dict, test_stat_dict


def compute_slice_wise_dose_scores(dose_kde_dict: Dict[str, List[float]]) -> List[float]:
    """Construct KDE score by summing up the different KDE fit values for a sample."""
    slice_wise_dose_scores = []
    # Loop over slices, dose_kde_values is a collection of all Dose_KDE values for a slice
    for dose_kde_values in zip(*{key: values for key, values in dose_kde_dict.items()
                                 if key in STATISTICS_FUNCTIONS.keys()}.values()):
        slice_wise_dose_scores.append(sum(dose_kde_values))
    return slice_wise_dose_scores


def full_pipeline_slice_wise_dose_scores(train_dataloader: DataLoader, test_dataloader: DataLoader,
                                         model: nn.Module, statistics: Tuple[str], max_n_batches: int = None,
                                         kde_func_dict: Dict[str, gaussian_kde] = None) -> Tuple[List[float],
                                                                                                 Dict[str, List[float]],
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
        test_stat_dict: a dictionary holding the corresponding statistics values ordered same way as in dose_kde_dict
    """
    # Fit training statistics if kde_func_dict with fits not given
    if kde_func_dict is None:
        fit_stat_dict = aggregate_slice_wise_statistics(model, train_dataloader, statistics, max_n_batches)
        kde_func_dict = fit_statistics(fit_stat_dict)

    dose_kde_dict, test_stat_dict = compute_slice_wise_dose_kde_scores(model, test_dataloader, kde_func_dict, statistics, max_n_batches)
    dose_scores = compute_slice_wise_dose_scores(dose_kde_dict)
    return dose_scores, dose_kde_dict, test_stat_dict

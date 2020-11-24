import sys
from collections import defaultdict

from scipy.stats.kde import gaussian_kde
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from uncertify.evaluation.inference import yield_inference_batches, BatchInferenceResult

from typing import List, Union, Dict


def aggregate_slice_wise_statistics(model: nn.Module, data_loader: DataLoader, statistics: List[str],
                                    max_n_batches: int = None, residual_threshold: float = None,
                                    health_state: str = 'all', only_non_empty: bool = True) -> dict:
    """Evaluate slice wise statistics and return aggregated results in a statistics-dict.

    Returns
        statistics_dict: will have a key for each statistic with a dictionary with 'all', 'healthy' and 'lesional'
                         keys with lists of slice-wise values for this statistic in this sub-dict
    """
    assert all([item in STATISTICS_FUNCTIONS for item in statistics]), f'Need to provide valid ' \
                                                                       f'statistics ({STATISTICS_FUNCTIONS})!'
    statistics_dict = defaultdict(list)
    for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches, residual_threshold,
                                                              progress_bar_suffix=f'(slice statistics '
                                                                                  f'{data_loader.dataset.name})')):
        if batch.slice_wise_is_lesional is not None:
            statistics_dict['is_lesional'].extend(list(batch.slice_wise_is_lesional))
        else:
            statistics_dict['is_lesional'].extend(np.zeros(len(batch.scan), dtype=bool))
        statistics_dict['is_empty'].extend(list(batch.slice_wise_is_empty))
        for statistic in statistics:
            statistics_dict[statistic].extend(get_slice_values_for_statistic(batch, statistic, health_state))
    if only_non_empty:
        non_empty_indices = np.invert(statistics_dict['is_empty'])
        statistics_dict = {key: list(np.array(values)[non_empty_indices]) for key, values in statistics_dict.items()}
    return statistics_dict


def get_slice_values_for_statistic(batch: BatchInferenceResult, statistic: str,
                                   health_state: str = 'all') -> np.ndarray:
    """Get the slice-wise value from a batch inference result for either all, only healthy or only lesional slices."""
    assert health_state in ['all', 'healthy', 'lesional'], f'Provided health_state argument {health_state} invalid.'
    assert statistic in STATISTICS_FUNCTIONS, f'Provided statistic not supported ' \
                                              f'(choose from {STATISTICS_FUNCTIONS.keys()}).'
    if health_state == 'lesional':
        if batch.segmentation is None:
            raise AttributeError(f'Requested lesional slice statistics but batch result has no ground truth'
                                 f'segmentation. Thus cannot decide which slices are lesional.')
    keep_indices = define_health_state_indices(health_state, batch)
    return STATISTICS_FUNCTIONS[statistic](batch)[keep_indices]


def define_health_state_indices(health_state: str, batch: BatchInferenceResult) -> np.ndarray:
    """Numpy array holding the indices of the slices to keep based on the health state."""
    if health_state == 'all':
        return np.ones(len(batch.scan), dtype=bool)
    elif health_state == 'healthy':
        return np.invert(batch.slice_wise_is_lesional)
    elif health_state == 'lesional':
        return batch.slice_wise_is_lesional
    else:
        raise ValueError(f'Health state {health_state} not supported.')


def fit_statistics(statistics_dict: Dict[str, List[float]]) -> Dict[str, gaussian_kde]:
    """For each set of inferred values for a given statistic, fit a KDE on it."""
    kde_func_dict = {stat_name: None for stat_name in statistics_dict.keys()}

    filtered_statistics_dict = {key: values for key, values in statistics_dict.items()
                                if key in STATISTICS_FUNCTIONS.keys()}

    for stat_name, values in filtered_statistics_dict.items():
        kde_func_dict[stat_name] = gaussian_kde(values)
    return kde_func_dict


"""
The following functions all adhere to the same function interface. They return batch-wise statistics as numpy arrays.
"""


def kl_div_batch_stat(batch: BatchInferenceResult) -> np.ndarray:
    return batch.kl_div


def rec_error_batch_stat(batch: BatchInferenceResult) -> np.ndarray:
    return batch.rec_err


def elbo_batch_stat(batch: BatchInferenceResult) -> np.ndarray:
    return -batch.kl_div + batch.rec_err


STATISTICS_FUNCTIONS = {
    'kl_div': kl_div_batch_stat,
    'rec_err': rec_error_batch_stat,
    'elbo': elbo_batch_stat
}


def statistics_dict_to_df(statistics_dict: dict, filter_out_empty: bool = True) -> pd.DataFrame:
    """Conversion function from statistics dict to pandas DataFrame."""
    df = pd.DataFrame.from_dict(statistics_dict)
    if filter_out_empty:
        df = df.loc[np.invert(df.is_empty)]
    return df

from collections import defaultdict

from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from uncertify.evaluation.inference import yield_inference_batches, BatchInferenceResult

from typing import List, Union


VALID_STATISTICS = ['kl_div', 'rec_err', 'elbo']


def get_slice_wise_statistics(model: nn.Module, data_loader: DataLoader, statistics: List[str],
                              max_n_batches: int = None, residual_threshold: float = None) -> dict:
    """Evaluate slice wise statistics and return aggregated results in a statistics-dict.

    Returns
        statistics_dict: will have a key for each statistic with a dictionary with 'all', 'healthy' and 'lesional'
                         keys with lists of slice-wise values for this statistic in this sub-dict
    """
    assert all([item in VALID_STATISTICS for item in statistics]), f'Need to provide valid ' \
                                                                   f'statistics ({VALID_STATISTICS})!'
    statistics_dict = {key: list() for key in statistics + ['is_lesional', 'is_empty']}
    for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches, residual_threshold,
                                                              progress_bar_suffix=f'Infer slice statistics')):
        statistics_dict['is_lesional'].extend(list(batch.slice_wise_is_lesional))
        statistics_dict['is_empty'].extend(list(batch.slice_wise_is_empty))
        for statistic in statistics:
            statistics_dict[statistic].extend(list(get_slice_values_for_statistic(batch, statistic, 'all')))
    return statistics_dict


def get_slice_values_for_statistic(batch: BatchInferenceResult, statistic: str,
                                   health_state: str = 'all') -> Union[np.ndarray]:
    """Get the slice-wise value from a batch inference result for either all, only healthy or only lesional slices."""
    assert health_state in ['all', 'healthy', 'lesional'], f'Provided health_state argument {health_state} invalid.'
    if health_state == 'lesional':
        if batch.segmentation is None:
            raise AttributeError(f'Requested lesional slice statistics but batch result has no ground truth'
                                 f'segmentation. Thus cannot decide which slices are lesional.')
    if statistic in ['kl_div', 'rec_err']:
        return get_statistic_values_from_batch(batch, statistic, health_state)
    else:
        raise NotImplementedError(f'This statistic {statistic} has no implementation for calculations yet.')


def get_statistic_values_from_batch(batch: BatchInferenceResult, statistic: str,
                                    health_state: str) -> np.ndarray:
    """Straight forward query if the result is already computed during inference."""
    assert statistic in ['kl_div', 'rec_err'], f'Provided statistic {statistic} not supported by this function.' \
                                               f'Need to calculate it somehow else.'
    if health_state == 'all':
        return getattr(batch, statistic)
    else:
        if health_state == 'healthy':
            return getattr(batch, statistic)[np.invert(batch.slice_wise_is_lesional)]
        elif health_state == 'lesional':
            return getattr(batch, statistic)[batch.slice_wise_is_lesional]


def statistics_dict_to_df(statistics_dict: dict) -> pd.DataFrame:
    """Conversion function."""
    statistics = list(statistics_dict.keys())

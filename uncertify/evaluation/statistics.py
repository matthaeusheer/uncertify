from collections import defaultdict
import math

from scipy.stats.kde import gaussian_kde
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from uncertify.evaluation.inference import yield_inference_batches, BatchInferenceResult
from uncertify.evaluation.waic import sample_wise_waic_scores
from uncertify.utils.custom_types import Tensor

from typing import List, Dict, Iterable


def aggregate_slice_wise_statistics(model: nn.Module, data_loader: DataLoader, statistics: Iterable[str],
                                    max_n_batches: int = None, residual_threshold: float = None,
                                    health_state: str = 'all', only_non_empty: bool = True,
                                    include_scans: bool = False) -> dict:
    """Evaluate slice wise statistics and return aggregated results in a statistics-dict.

    Returns
        statistics_dict: will have a key for each statistic with a dictionary with 'all', 'healthy' and 'lesional'
                         keys with lists of slice-wise values for this statistic in this sub-dict
    """
    assert all([item in STATISTICS_FUNCTIONS for item in statistics]), f'Need to provide valid ' \
                                                                       f'statistics ({STATISTICS_FUNCTIONS})!'
    statistics_dict = defaultdict(list)  # statistics are keys and list of scores are values
    slice_wise_scans = []  # track scans for later visualization
    slices_keep_mask = []  # the indices mask which decides which slices we keep
    slice_wise_seg_maps = []
    for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches, residual_threshold,
                                                              progress_bar_suffix=f'(slice statistics '
                                                                                  f'{data_loader.dataset.name})')):
        batch_size, _, _, _ = batch.scan.shape
        # Track which slices we keep based on slice health state and emptiness of brain mask
        health_state_mask = define_health_state_mask(health_state, batch)
        is_not_empty_mask = np.invert(batch.slice_wise_is_empty)
        batch_keep_mask = np.logical_and(health_state_mask, is_not_empty_mask)
        slices_keep_mask.extend(list(batch_keep_mask))

        # Track lesional slices
        is_lesional = list(batch.slice_wise_is_lesional) if batch.slice_wise_is_lesional is not None \
            else list(np.zeros(len(batch.scan), dtype=bool))
        statistics_dict['is_lesional'].extend(is_lesional)

        # Add the actual statistic
        for statistic in statistics:
            statistics_dict[statistic].extend(list(STATISTICS_FUNCTIONS[statistic](batch)))

        # Track scans and potentially ground truth segmentation for visualizations later on
        for scan in batch.scan:
            slice_wise_scans.append(scan)
        if batch.segmentation is not None:
            for seg in batch.segmentation:
                slice_wise_seg_maps.append(seg)
        else:
            for _ in range(batch_size):
                slice_wise_seg_maps.append(torch.zeros_like(batch.scan[0]))
    # Apply indices mask to filter out empty slices or slices from other health state
    mask_slice_indices = [idx for idx, keep_slice in enumerate(slices_keep_mask) if keep_slice]
    statistics_dict = {key: [values[idx] for idx in mask_slice_indices] for key, values in statistics_dict.items()}

    slice_wise_scans = [slice_wise_scans[idx] for idx in mask_slice_indices]
    slice_wise_seg_maps = [slice_wise_seg_maps[idx] for idx in mask_slice_indices]
    statistics_dict.update({'scans': slice_wise_scans})
    statistics_dict.update({'segmentations': slice_wise_seg_maps})
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
    keep_indices = define_health_state_mask(health_state, batch)
    return STATISTICS_FUNCTIONS[statistic](batch)[keep_indices]


def define_health_state_mask(health_state: str, batch: BatchInferenceResult) -> np.ndarray:
    """Numpy array holding the True / False for the slices to keep based on the health state."""
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
    kde_func_dict = {stat_name: None for stat_name in statistics_dict.keys()
                     if stat_name in STATISTICS_FUNCTIONS.keys()}

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


def rec_error_entropy_batch_stat(batch: BatchInferenceResult) -> np.ndarray:
    """Calculates the slice-wise entropy of normalized l1 residual images."""
    # Residual = residual_l1_max
    residual_batch = batch.residual
    batch_size, _, _, _ = residual_batch.shape
    # Normalization
    slice_wise_sum = torch.sum(residual_batch, dim=(1, 2, 3))
    for slice_idx in range(batch_size):
        residual_batch[slice_idx] /= slice_wise_sum[slice_idx]
    # Calculate entropy per slice
    entropy_array = np.empty(batch_size)
    for idx, image in enumerate(residual_batch):
        entropy_array[idx] = get_entropy(image[0])
    return entropy_array


def waic_batch_stat(batch: BatchInferenceResult) -> np.ndarray:
    raise NotImplementedError(f'WAIC score statistics not implemented yet!')


STATISTICS_FUNCTIONS = {
    'kl_div': kl_div_batch_stat,
    'rec_err': rec_error_batch_stat,
    'elbo': elbo_batch_stat,
    'entropy': rec_error_entropy_batch_stat
}


def get_entropy(image: Tensor) -> float:
    """Calculate the standard entropy by considering all pixel values as a probability distribution."""
    entropy = 0.0
    for pix in image[image > 0.0].flatten():
        entropy -= float(pix) * math.log2(float(pix))
    return entropy


def statistics_dict_to_df(statistics_dict: dict, filter_out_empty: bool = True) -> pd.DataFrame:
    """Conversion function from statistics dict to pandas DataFrame."""
    df = pd.DataFrame.from_dict(statistics_dict)
    if filter_out_empty:
        df = df.loc[np.invert(df.is_empty)]
    return df

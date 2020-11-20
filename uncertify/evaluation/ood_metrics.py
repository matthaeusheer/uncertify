import logging
from collections import defaultdict
from math import pow

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.utils.custom_types import Tensor

from typing import Iterable, List, Tuple, Optional

N_ABNORMAL_PIXELS_THRESHOLD_LESIONAL = 20  # Sample with more abnormal pixels are considered lesional

LOG = logging.getLogger(__name__)


def waic_score_per_sample(log_likelihoods: Iterable[float]) -> float:
    """Calculates the WAIC score for a single sample based on log_likelihoods which come from different models.

    Arguments:
        log_likelihoods: a list of log likelihoods coming from different ensemble models to calculate WAIC for
    """
    mean_log_likelihood = float(np.mean(log_likelihoods))
    var_log_likelihood = float(np.var(log_likelihoods))
    return mean_log_likelihood - pow(var_log_likelihood, 2)


def sample_wise_waic_scores(models: Iterable[nn.Module], data_loader: DataLoader,
                            residual_threshold: float = None, max_n_batches: int = None,
                            return_slices: bool = False) -> Tuple[List[float],
                                                                  List[bool],
                                                                  Optional[List[Tensor]]]:
    """Computes all per-slice WAIC scores for all batches of the generator.

    Arguments:
        models: an iterable of trained ensemble models
        data_loader: the pytorch dataloader to receive the data from
        residual_threshold: threshold in the residual image to calculate mark abnormal pixels  # TODO: Needed???
        max_n_batches: limit number of batches used in analysis, handy for debugging
        return_slices: whether to aggregate and return the individual slices (should be turned of for large evaluation)
    Returns:
        slice_wise_waic_scores: a list of waic scores, one for each slice, so the size is ~(num_batches * batch_size)
        slice_wise_is_lesional: a list of True (for lesional) or False (for normal) values, one for each slice
        slice_wise_scans [Optional]: a list of scan tensors for further analysis connected to the slice_wise_waic_scores
    """
    # Keys are slice indices, values are a list of log likelihoods coming from different models
    slice_wise_log_likelihoods = defaultdict(list)
    # A list holding information for every slice if it's lesional (True) or normal (False)
    slice_wise_is_lesional = []
    # A list of pytorch tensors holding a scan of one slice
    slice_wise_scans = []

    global_slice_idx = 0
    for model_idx, model in enumerate(models):  # will yield same input data for every ensemble model
        for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches, residual_threshold,
                                                                  progress_bar_suffix=f'WAIC (ensemble {model_idx})')):
            # Used to exclude slices which have an empty mask, i.e. no actual scan
            per_slice_log_likelihoods = -batch.kl_div + batch.rec_err
            for slice_idx, log_likelihood in enumerate(per_slice_log_likelihoods):
                if not batch.slice_wise_is_empty[slice_idx]:
                    slice_wise_log_likelihoods[global_slice_idx].append(log_likelihood)
                    if model_idx == 0:
                        if batch.segmentation is not None:
                            n_abnormal_pixels = float(torch.sum(batch.segmentation[slice_idx] > 0))
                            slice_wise_is_lesional.append(n_abnormal_pixels > N_ABNORMAL_PIXELS_THRESHOLD_LESIONAL)
                        else:
                            slice_wise_is_lesional.append(False)
                        if return_slices:
                            slice_wise_scans.append(batch.scan[slice_idx])
                    # Increase global slice counter when we added a slice to the evaluation list
                    global_slice_idx += 1
        # Reset the global slice counter when iterating over batches and slices using the next ensemble model
        global_slice_idx = 0

    # Now loop over all lists of likelihood values (one list per slice) and compute the WAIC score
    slice_wise_ood_scores = []
    for log_likelihoods in slice_wise_log_likelihoods.values():
        mean = float(np.mean(log_likelihoods))
        var = float(np.var(log_likelihoods))
        waic = mean - var
        slice_wise_ood_scores.append(waic)

    return slice_wise_ood_scores, slice_wise_is_lesional, slice_wise_scans if len(slice_wise_scans) > 0 else None

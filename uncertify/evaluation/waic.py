import logging
from collections import defaultdict, namedtuple

import torch
from typing import Iterable, Tuple, List, Optional

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from uncertify.evaluation.inference import yield_inference_batches, BatchInferenceResult
from uncertify.utils.custom_types import Tensor

N_ABNORMAL_PIXELS_THRESHOLD_LESIONAL = 20  # Sample with more abnormal pixels are considered lesional
LOG = logging.getLogger(__name__)

ReturnTuple = namedtuple('OodScores', ['slice_wise_waic_scores', 'slice_wise_is_lesional',
                                       'slice_wise_scans', 'slice_wise_elbos', 'slice_wise_kl_div',
                                       'slice_wise_rec_err'])


def sample_wise_waic_scores(models: Iterable[nn.Module], data_loader: DataLoader, max_n_batches: int = None,
                            return_slices: bool = False) -> ReturnTuple:
    """Computes all per-slice WAIC scores for all batches of the generator as well as the ELBO for one model.

    Arguments:
        models: an iterable of trained ensemble models
        data_loader: the pytorch dataloader to receive the data from
        max_n_batches: limit number of batches used in analysis, handy for debugging
        return_slices: whether to aggregate and return the individual slices (should be turned of for large evaluation)
    Returns:
        slice_wise_waic_scores: a list of waic scores, one for each slice, so the size is ~(num_batches * batch_size)
        slice_wise_is_lesional: a list of True (for lesional) or False (for normal) values, one for each slice
        slice_wise_scans [Optional]: a list of scan tensors for further analysis connected to the slice_wise_waic_scores
    """
    LOG.info(f'Getting slice-wise WAIC scores for {data_loader.dataset.name}')
    # Keys are slice indices, values are a list of log likelihoods coming from different models
    slice_wise_elbos_ensemble = defaultdict(list)
    # A list holding information for every slice if it's lesional (True) or normal (False)
    slice_wise_is_lesional = []
    # A list of pytorch tensors holding a scan of one slice
    slice_wise_scans = []
    # The slice-wise ELBO evaluated on the first of the ensembles
    slice_wise_elbo_scores = []
    # The slice-wise KL Divergence evaluated on the first of the ensembles
    slice_wise_kl_div = []
    # The slice-wise reconstruction error evaluated on the first of the ensembles
    slice_wise_rec_err = []

    global_slice_idx = 0
    for model_idx, model in enumerate(models):  # will yield same input data for every ensemble model
        for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches,
                                                                  progress_bar_suffix=f'WAIC (ensemble {model_idx})')):
            slice_wise_elbos = batch.rec_err - batch.kl_div
            for slice_idx, slice_elbo in enumerate(slice_wise_elbos):
                if not batch.slice_wise_is_empty[slice_idx]:
                    slice_wise_elbos_ensemble[global_slice_idx].append(slice_elbo)
                    if model_idx == 0:
                        slice_wise_elbo_scores.append(slice_wise_elbos[slice_idx])
                        slice_wise_kl_div.append(batch.kl_div[slice_idx])
                        slice_wise_rec_err.append(batch.rec_err[slice_idx])
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

    # Now loop over all lists of elbo values (one list per slice) and compute the WAIC score
    slice_wise_waic_scores = []
    for slice_elbo_lists in slice_wise_elbos_ensemble.values():
        mean = float(np.mean(slice_elbo_lists))
        var = float(np.var(slice_elbo_lists))
        waic = (mean - var)
        slice_wise_waic_scores.append(waic)

    slice_wise_scans = slice_wise_scans if len(slice_wise_scans) > 0 else None

    return ReturnTuple(slice_wise_waic_scores,
                       slice_wise_is_lesional,
                       slice_wise_scans,
                       slice_wise_elbo_scores,
                       slice_wise_kl_div,
                       slice_wise_rec_err)

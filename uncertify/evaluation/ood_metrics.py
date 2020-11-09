from pathlib import Path
from collections import defaultdict
from math import pow

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.models.vae import load_vae_baur_model

from typing import Iterable, List


def waic_score_per_sample(log_likelihoods: Iterable[float]) -> float:
    """Calculates the WAIC score for a single sample based on log_likelihoods which come from different models.

    Arguments:
        log_likelihoods: a list of log likelihoods coming from different ensemble models to calculate WAIC for
    """
    mean_log_likelihood = float(np.mean(log_likelihoods))
    var_log_likelihood = float(np.var(log_likelihoods))
    return mean_log_likelihood - pow(var_log_likelihood, 2)


def sample_wise_waic_scores(models: Iterable[nn.Module], data_loader: DataLoader,
                            residual_threshold: float = None, max_n_batches: int = None) -> List[float]:
    """Computes all per-slice WAIC scores for all batches of the generator.

    Arguments:
        models: an iterable of trained ensemble models
        data_loader: the pytorch dataloader to receive the data from
        residual_threshold: threshold in the residual image to calculate mark abnormal pixels  # TODO: Needed???
        max_n_batches: limit number of batches used in analysis, handy for debugging
    Returns:
        slice_wise_waic_scores: a list of waic scores, one for each slice, so the size is ~(num_batches * batch_size)
    """
    slice_wise_log_likelihoods = defaultdict(list)
    batch_size = data_loader.batch_size

    for model_idx, model in enumerate(models):  # will yield same input data for every ensemble model
        for batch_idx, batch in enumerate(yield_inference_batches(data_loader, model, max_n_batches, residual_threshold,
                                                                  progress_bar_suffix=f'WAIC (ensemble {model_idx})')):
            per_slice_log_likelihoods = -batch.kl_div + batch.rec_err
            for slice_idx, log_likelihood in enumerate(per_slice_log_likelihoods):
                slice_wise_log_likelihoods[batch_idx * batch_size + slice_idx].append(log_likelihood)

    slice_wise_waic_scores = []
    for log_likelihoods in slice_wise_log_likelihoods.values():
        mean = float(np.mean(log_likelihoods))
        var = float(np.var(log_likelihoods))
        waic = mean - var
        slice_wise_waic_scores.append(waic)

    return slice_wise_waic_scores


def load_ensemble_models(dir_path: Path, file_names: List[str], model_type: str = 'vae_baur') -> List[nn.Module]:
    assert model_type == 'vae_baur', f'No other model is defined for loading ensemble methods yet.'
    models = []
    for name in file_names:
        models.append(load_vae_baur_model(dir_path / name))
    return models

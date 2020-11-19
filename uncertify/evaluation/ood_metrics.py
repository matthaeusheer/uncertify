import logging
from pathlib import Path
from collections import defaultdict
from math import pow

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from uncertify.evaluation.inference import yield_inference_batches
from uncertify.models.vae import load_vae_baur_model
from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult, \
    PixelAnomalyDetectionResult, SliceAnomalyDetectionResults, OODDetectionResults
from uncertify.evaluation.evaluation_pipeline import run_ood_detection_performance, print_results_from_evaluation_dirs
from uncertify.utils.custom_types import Tensor
from uncertify.common import DATA_DIR_PATH

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
            slice_wise_is_empty = torch.sum(batch.mask == True, axis=(1, 2, 3)).numpy() == 0
            per_slice_log_likelihoods = -batch.kl_div + batch.rec_err
            for slice_idx, log_likelihood in enumerate(per_slice_log_likelihoods):
                if not slice_wise_is_empty[slice_idx]:
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


def run_ood_evaluations(train_dataloader: DataLoader, dataloader_dict: dict, ensemble_models: List[nn.Module],
                        residual_threshold: float = None, max_n_batches: int = None,
                        eval_results_dir: Path = DATA_DIR_PATH / 'evaluation',
                        print_results_when_done: bool = True, ) -> None:
    """Run OOD evaluation pipeline for multiple dataloaders amd store output in evaluation output directory.

    Arguments
        train_dataloader: the in-distribution dataloader from training the model
        dataloader_dict: (name, dataloader) dictionary
        ensemble_models: a list of trained ensemble models
        residual_threshold: as usual
        eval_results_dir: for each dataloader, i.e. each run an output folder with results will be created here
        print_results_when_done: self-explanatory
    """
    run_numbers = []
    for name, dataloader in dataloader_dict.items():
        LOG.info(f'OOD evaluation for {name}...')
        eval_cfg = EvaluationConfig()
        eval_cfg.do_plots = True
        eval_cfg.use_n_batches = max_n_batches
        results = EvaluationResult(eval_results_dir, eval_cfg, PixelAnomalyDetectionResult(),
                                   SliceAnomalyDetectionResults(), OODDetectionResults())
        results.pixel_anomaly_result.best_threshold = residual_threshold
        results.make_dirs()
        run_numbers.append(results.run_number)
        run_ood_detection_performance(ensemble_models, train_dataloader, dataloader, eval_cfg, results)
        results.test_set_name = name
        results.to_json()
    if print_results_when_done:
        print_results_from_evaluation_dirs(eval_results_dir, run_numbers, print_results_only=True)


def run_ood_to_ood_dict(dataloader_dict: dict, ensemble_models: list, num_batches: int = None) -> dict:
    """Run OOD detection and organise output such that healthy vs. lesional analysis can be performed.

    # TODO: Define an interface for functions that return slice-wise ood scores and take the function as
            an input parameter s.t. this works for different OOD metrics.
    Returns
        ood_dict: one sub-dict for every dataloader where the name is the key and for the sub-dicts the
                  keys are ['all', 'healthy', 'lesional', 'healthy_scans', 'lesional_scans' where the values for the
                  scans keys are slice-wise pytorch tensors and the rest are actual OOD scores in a list either
                  all of them, only the healthy ones or only lesional ones
    """
    ood_dict = {}
    for name, data_loader in dataloader_dict.items():
        LOG.info(f'WAIC score calculation for {name} ({num_batches * data_loader.batch_size} slices)...')
        slice_wise_waic_scores, slice_wise_is_lesional, scans = sample_wise_waic_scores(models=ensemble_models,
                                                                                        data_loader=data_loader,
                                                                                        max_n_batches=num_batches,
                                                                                        return_slices=True)
        # Organise as healthy / unhealthy
        healthy_scores = []
        lesional_scores = []
        healthy_scans = []
        lesional_scans = []
        for idx in range(len(slice_wise_waic_scores)):
            is_lesional_slice = slice_wise_is_lesional[idx]
            if is_lesional_slice:
                lesional_scores.append(slice_wise_waic_scores[idx])
                if scans is not None:
                    lesional_scans.append(scans[idx])
            else:
                healthy_scores.append(slice_wise_waic_scores[idx])
                if scans is not None:
                    healthy_scans.append(scans[idx])

        dataset_ood_dict = {'all': slice_wise_waic_scores, 'healthy': healthy_scores, 'lesional': lesional_scores,
                            'healthy_scans': healthy_scans, 'lesional_scans': lesional_scans}
        ood_dict[name] = dataset_ood_dict
    return ood_dict


def load_ensemble_models(dir_path: Path, file_names: List[str], model_type: str = 'vae_baur') -> List[nn.Module]:
    assert model_type == 'vae_baur', f'No other model is defined for loading ensemble methods yet.'
    models = []
    for name in file_names:
        models.append(load_vae_baur_model(dir_path / name))
    return models

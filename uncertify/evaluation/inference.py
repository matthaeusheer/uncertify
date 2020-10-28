import logging
import itertools
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from uncertify.evaluation.utils import threshold_batch_to_one_zero, residual_l1_max, convert_segmentation_to_one_zero
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.utils.custom_types import Tensor

from typing import Generator, Callable, Iterable, Tuple, Optional

LOG = logging.getLogger(__name__)


@dataclass
class PerPixelResults:
    pass


@dataclass
class PerSampleResults:
    dice_per_patient_mean: float = None
    dice_per_patient_std: float = None
    iou_per_patient_mean: float = None
    iou_per_patient_std: float = None


@dataclass
class MultiThreshAnalysis:
    """Performance behaviour for different residual thresholds instead of only looking at the one determined."""
    thresholds: Iterable[float] = None
    dice_scores_mean: Iterable[float] = None
    dice_scores_std: Iterable[float] = None


@dataclass
class OodResults:
    pass


@dataclass
class InferenceResults:
    accepted_fpr: float
    residual_threshold: float
    per_pixel_results: PerPixelResults
    per_sample_results: PerSampleResults
    ood_results: OodResults


@dataclass
class BatchInferenceResult:
    """The raw per-batch result we can infer from running a single batch through the network including thresholding."""
    scan: Tensor = None
    mask: Tensor = None
    segmentation: Tensor = None
    reconstruction: Tensor = None
    residual: Tensor = None
    residuals_thresholded: Tensor = None
    residual_threshold: float = None
    total_loss: float = None
    mean_kld_div: float = None
    mean_rec_err: float = None
    kl_div: Tensor = None
    rec_err: Tensor = None
    latent_code: Tensor = None


def yield_inference_batches(data_loader: DataLoader,
                            trained_model: torch.nn.Module,
                            max_batches: int = None,
                            residual_threshold: float = None,
                            residual_fn: Callable = residual_l1_max,
                            get_batch_fn: Callable = lambda batch: batch['scan'],
                            progress_bar_suffix: str = '') -> Generator[BatchInferenceResult, None, None]:
    """Run inference on batches from a data loader on a trained model.

    Arguments:
        data_loader: a dataloader
        trained_model: a trained VAE model instance
        max_batches: possibility to take only max_batches first batches, handy for debugging or speedy dev work
        residual_threshold: pixels with a higher value than this in the residual image are marked as abnormal
        residual_fn: function defining the way we define the residual image / batch
        get_batch_fn: function to get the input (scan) batch tensor from the input batch of the dataloader
        progress_bar_suffix: possibility to add some informative stuff to the progress bar during inference

    Returns:
        result: a BatchInferenceResult dataclass container holding all results from the batch inference
    """
    result = BatchInferenceResult()

    data_generator = itertools.islice(data_loader, max_batches) if max_batches is not None else data_loader
    n_batches = max_batches if max_batches is not None else len(data_loader)
    for batch in tqdm(data_generator, desc=f'Inferring batches {progress_bar_suffix}', initial=1, total=n_batches):
        scan_batch = get_batch_fn(batch)

        # Run actual inference for batch
        inference_result = trained_model(scan_batch)
        rec_batch, mu, log_var, total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err, latent_code = inference_result

        # Do residual calculation
        residual_batch = residual_fn(rec_batch, scan_batch)

        # Fill output container
        result.scan = scan_batch
        result.reconstruction = rec_batch
        result.residual = residual_batch
        result.latent_code = latent_code

        if residual_threshold is not None:
            result.residual_threshold = residual_threshold
            result.residuals_thresholded = threshold_batch_to_one_zero(residual_batch, residual_threshold)

        if 'seg' in batch.keys():
            result.segmentation = convert_segmentation_to_one_zero(batch['seg'])

        if 'mask' in batch.keys():
            result.mask = batch['mask']

        for key, value in zip(['total_loss', 'mean_kld_div', 'mean_rec_err', 'kl_div', 'rec_err'],
                              [total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err]):
            result.__setattr__(key, value)

        yield result


def yield_y_true_y_pred(data_loader: DataLoader,
                        trained_model: torch.nn.Module,
                        max_n_batches: int = None,
                        residual_threshold: float = None,
                        residual_fn: Callable = residual_l1_max) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Yield flattened vectors over all (of max_n_batches, if not None) batches for y_true and y_pred.

    For Args: see yield_reconstruction_batches. Similar.

    Returns:
        a tuple of (y_true, anomaly_score) where the two entries are numpy arrays representing the true anomaly label
        and the anomaly score (residual value) for each pixel within the brain mask
    """
    y_true_batch_list = []
    anomaly_score_batch_list = []
    y_pred_batch_list = []

    for batch_idx, batch in enumerate(yield_inference_batches(data_loader=data_loader,
                                                              trained_model=trained_model,
                                                              max_batches=max_n_batches,
                                                              residual_threshold=residual_threshold,
                                                              residual_fn=residual_fn)):
        with torch.no_grad():
            # Get ground truth and anomaly score / residual (higher residual means higher anomaly score)
            batch_y_true = batch.segmentation[batch.mask].flatten().numpy()
            y_true_batch_list.append(batch_y_true)

            anomaly_score = batch.residual[batch.mask].flatten().numpy()
            anomaly_score_batch_list.append(anomaly_score)

            if residual_threshold is not None:
                batch_y_pred = batch.residuals_thresholded[batch.mask].flatten().numpy()
                y_pred_batch_list.append(batch_y_pred)

    # Concatenate ground truths and scores over all batches
    y_true = np.hstack(y_true_batch_list)
    y_pred_proba = np.hstack(anomaly_score_batch_list)
    y_pred = np.hstack(y_pred_batch_list) if len(y_pred_batch_list) > 0 else None

    return y_true, y_pred_proba, y_pred


def infer_latent_space_samples(model: torch.nn.Module, latent_samples: Tensor) -> Tensor:
    """Run inference only on the decoder part of the model using some samples from the latent space."""
    with torch.no_grad():
        return model._decoder(latent_samples)

import logging
import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uncertify.evaluation.utils import threshold_batch_to_one_zero, convert_segmentation_to_one_zero
from uncertify.evaluation.utils import residual_l1, residual_l1_max
from uncertify.evaluation.median_pool import MedianPool2d
from uncertify.evaluation.post_processing import BinaryErosion
from uncertify.utils.custom_types import Tensor

from typing import Generator, Callable, List, Dict

LOG = logging.getLogger(__name__)


@dataclass
class BatchInferenceResult:
    """The raw per-batch result we can infer from running a single batch through the network including thresholding."""
    # Image-like torch tensors
    scan: Tensor = None
    mask: Tensor = None
    segmentation: Tensor = None
    reconstruction: Tensor = None
    residual: Tensor = None
    residuals_thresholded: Tensor = None
    # Numpy arrays or single values
    residual_threshold: float = None
    slice_wise_is_empty: List[bool] = None
    slice_wise_is_lesional: np.ndarray = None
    total_loss: float = None
    mean_kld_div: float = None
    mean_rec_err: float = None
    kl_div: np.ndarray = None
    rec_err: np.ndarray = None
    latent_code: np.ndarray = None


@dataclass
class AnomalyScores:
    y_true: List[float] = field(default_factory=list)
    y_pred_proba: List[float] = field(default_factory=list)
    y_pred: List[float] = field(default_factory=list)


class SliceWiseCriteria(Enum):
    """Anomaly score criteria which can be used for slice-wise anomaly detection."""
    REC_TERM = 1
    KL_TERM = 2
    ELBO = 3


@dataclass
class SliceWiseAnomalyScores:
    criteria: SliceWiseCriteria
    anomaly_score: AnomalyScores


@dataclass
class AnomalyInferenceScores:
    """Container for pixel-wise and slice-wise anomaly detection predictions."""
    pixel_wise: AnomalyScores = field(default_factory=list)
    slice_wise: Dict[str, SliceWiseAnomalyScores] = field(default_factory=dict)


N_PIXEL_LESIONAL_THRESHOLD = 50


def yield_inference_batches(data_loader: DataLoader,
                            trained_model: torch.nn.Module,
                            max_batches: int = None,
                            residual_threshold: float = None,
                            residual_fn: Callable = residual_l1_max,
                            get_batch_fn: Callable = lambda batch: batch['scan'],
                            median_filter: MedianPool2d = None,  # MedianPool2d(kernel_size=5, padding=2),
                            mask_eroder: BinaryErosion = None,  # BinaryErosion(num_iterations=3),
                            progress_bar_suffix: str = '',
                            manual_seed_val: int = None,
                            use_mask: bool = True) -> Generator[BatchInferenceResult, None, None]:
    """Run inference on batches from a data loader on a trained model.

    Arguments:
        data_loader: a dataloader
        trained_model: a trained VAE model instance
        max_batches: possibility to take only max_batches first batches, handy for debugging or speedy dev work
        residual_threshold: pixels with a higher value than this in the residual image are marked as abnormal
        residual_fn: function defining the way we define the residual image / batch
        get_batch_fn: function to get the input (scan) batch tensor from the input batch of the dataloader
        median_filter: if filter provided, does smooth the residual map
        mask_eroder: erode brain mask inwards to get rid of false positive pixel anomalies on the brain edges
        progress_bar_suffix: possibility to add some informative stuff to the progress bar during inference
        manual_seed_val: resets the random seed for torch
        use_mask: if True, will use the mask during inference, that is the loss is computed only on the masked region

    Returns:
        result: a BatchInferenceResult dataclass container holding all results from the batch inference
    """
    result = BatchInferenceResult()

    if manual_seed_val is not None:
        torch.manual_seed(manual_seed_val)

    data_generator = itertools.islice(data_loader, max_batches) if max_batches is not None else data_loader
    n_batches = max_batches if max_batches is not None else len(data_loader)
    for batch in tqdm(data_generator, desc=f'Inferring batches {progress_bar_suffix}', initial=1, total=n_batches):
        scan_batch = get_batch_fn(batch)

        # Run actual inference for batch
        trained_model.eval()
        with torch.no_grad():
            if use_mask:
                potential_mask = batch['mask'] if 'mask' in batch.keys() else None
            else:
                potential_mask = None
            inference_result = trained_model(scan_batch, potential_mask)
        rec_batch, mu, log_var, total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err, latent_code = inference_result

        if torch.isnan(rec_batch).any():
            raise RuntimeError(f'Reconstruction contains nan values!')

        if 'mask' in batch.keys():
            mask_batch = batch['mask']
            result.mask = mask_batch
            result.slice_wise_is_empty = torch.sum(mask_batch, axis=(1, 2, 3)).numpy() == 0
        else:
            mask_batch = torch.ones_like(batch['scan']) > 0
            result.slice_wise_is_empty = np.zeros(len(mask_batch), dtype=bool)

        if 'seg' in batch.keys():
            result.segmentation = convert_segmentation_to_one_zero(batch['seg'])
            result.slice_wise_is_lesional = (torch.sum(result.segmentation, axis=(1, 2, 3)) >
                                             N_PIXEL_LESIONAL_THRESHOLD).numpy()
        else:
            result.slice_wise_is_lesional = np.zeros(len(batch['scan']), dtype=bool)

        residual_batch = residual_fn(rec_batch, scan_batch)
        if mask_eroder is not None:
            residual_batch *= mask_eroder(mask_batch.numpy())

        if median_filter is not None:
            with torch.no_grad():
                residual_batch = median_filter(residual_batch.cuda()).cpu()

        # Fill output container
        result.scan = scan_batch
        result.reconstruction = rec_batch
        result.residual = residual_batch
        result.latent_code = latent_code

        if residual_threshold is not None:
            result.residual_threshold = residual_threshold
            result.residuals_thresholded = threshold_batch_to_one_zero(residual_batch, residual_threshold)

        for key, value in zip(['total_loss', 'mean_kld_div', 'mean_rec_err', 'kl_div', 'rec_err'],
                              [total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err]):
            if len(value.shape) == 0:  # single values
                value = float(value)
            else:
                value = value.detach().numpy()
            result.__setattr__(key, value)

        yield result


def yield_anomaly_predictions(data_loader: DataLoader,
                              trained_model: torch.nn.Module,
                              max_n_batches: int = None,
                              residual_threshold: float = None,
                              residual_fn: Callable = residual_l1_max,
                              use_mask: bool = True) -> AnomalyInferenceScores:
    """Computes and yields AnomalyInferenceScores, that is, pixel- and slice-wise anomaly prediction scores.

    For Args: see yield_reconstruction_batches. Similar.

    Returns:
        anomaly_scores: an AnomalyInferenceScores object which holds pixel- and slice-wise anomaly scores
    """
    # Initially empty scores objects to fill up subsequently
    anomaly_scores = AnomalyInferenceScores(pixel_wise=AnomalyScores(),
                                            slice_wise={criteria.name: SliceWiseAnomalyScores(criteria, AnomalyScores())
                                                        for criteria in SliceWiseCriteria})

    for batch_idx, batch in enumerate(yield_inference_batches(data_loader=data_loader,
                                                              trained_model=trained_model,
                                                              max_batches=max_n_batches,
                                                              residual_threshold=residual_threshold,
                                                              residual_fn=residual_fn,
                                                              use_mask=use_mask)):
        with torch.no_grad():
            # Step 1 - Pixel-wise
            # Get ground truth and anomaly score / residual (higher residual means higher anomaly score)
            batch_y_true = batch.segmentation[batch.mask].flatten().numpy()
            anomaly_scores.pixel_wise.y_true.extend(batch_y_true)

            batch_y_pred_proba = batch.residual[batch.mask].flatten().numpy()
            anomaly_scores.pixel_wise.y_pred_proba.extend(batch_y_pred_proba)

            if residual_threshold is not None:
                batch_y_pred = batch.residuals_thresholded[batch.mask].flatten().numpy()
                anomaly_scores.pixel_wise.y_pred.extend(batch_y_pred)

            # Step 2 - Slice-wise
            # Treat the loss term components as anomaly scores
            slice_wise_elbo = list(-batch.kl_div + batch.rec_err)
            slice_wise_kl_div = list(batch.kl_div)
            slice_wise_rec_error = list(batch.rec_err)
            slice_wise_is_empty = list(batch.slice_wise_is_empty)
            slice_wise_n_lesional_pixels = torch.sum(batch.segmentation > 0, axis=(1, 2, 3)).numpy()
            slice_wise_is_lesional = [count > N_PIXEL_LESIONAL_THRESHOLD for count in slice_wise_n_lesional_pixels]
            for slice_idx, (is_empty, is_lesional, kl_div, rec_err, elbo) in enumerate(zip(slice_wise_is_empty,
                                                                                           slice_wise_is_lesional,
                                                                                           slice_wise_kl_div,
                                                                                           slice_wise_rec_error,
                                                                                           slice_wise_elbo)):
                if is_empty:
                    continue

                # y_pred_proba
                anomaly_scores.slice_wise['KL_TERM'].anomaly_score.y_pred_proba.append(kl_div)
                anomaly_scores.slice_wise['REC_TERM'].anomaly_score.y_pred_proba.append(rec_err)
                anomaly_scores.slice_wise['ELBO'].anomaly_score.y_pred_proba.append(elbo)

                # y_true
                for criteria in SliceWiseCriteria:
                    anomaly_scores.slice_wise[criteria.name].anomaly_score.y_true.append(is_lesional)
    return anomaly_scores

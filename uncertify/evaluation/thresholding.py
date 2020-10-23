import itertools
import logging

import numpy as np
import torch

from uncertify.data.dataloaders import DataLoader
from uncertify.deploy import yield_reconstructed_batches
from uncertify.metrics.classification import false_positive_rate

from typing import Tuple, List, Iterable

LOG = logging.getLogger(__name__)


def threshold_vs_fpr(data_loader: DataLoader, model: torch.nn.Module, thresholds: Iterable = None,
                     use_ground_truth: bool = False, n_batches_per_thresh: int = 50) -> Tuple[List, List]:
    """Calculate the false positive rate """
    if thresholds is None:
        thresholds = list(np.linspace(0.1, 1, 10))

    false_positive_rates = []
    for threshold in thresholds:
        false_positive_rates.append(calculate_mean_false_positive_rate(threshold, data_loader, model,
                                                                       use_ground_truth, n_batches_per_thresh))
    return thresholds, false_positive_rates


def calculate_mean_false_positive_rate(threshold: float, data_loader: DataLoader, model: torch.nn.Module,
                                       use_ground_truth: bool = False, n_batches_per_thresh: int = None) -> float:
    """Calculate the mean false positive rate (mean taken over batches) for a given threshold.

    If use_ground_truth=True, the data_loader must provide tensors with ground truth segmentation under "seg" key.
    Else, every "anomaly" pixel in the thresholded residual is considered an outlier. In this setting, only healthy
    samples (i.e. from the training data) should be used to make the assumption hold.
    """
    result_generator = yield_reconstructed_batches(data_loader, model,
                                                   residual_threshold=threshold,
                                                   max_batches=n_batches_per_thresh,
                                                   print_statistics=False)
    per_batch_fpr = []
    if n_batches_per_thresh is not None:
        result_generator = itertools.islice(result_generator, n_batches_per_thresh)
    with torch.no_grad():
        for batch_idx, batch in enumerate(result_generator):
            prediction = batch['thresh'][batch['mask']]
            pred_np = prediction.numpy().astype(int)
            if use_ground_truth:
                try:
                    ground_truth = batch['seg'][batch['mask']]
                except KeyError:
                    LOG.exception(f'When use_ground_truth=True, the data_loader must '
                                  f'provide batches under "seg" key. Exit.')
                    raise
                gt_np = ground_truth.numpy().astype(int)
                fpr = false_positive_rate(pred_np, gt_np)
            else:
                fpr = false_positive_rate(pred_np, np.zeros_like(pred_np))
            per_batch_fpr.append(fpr)
            if (batch_idx + 1) % 50 == 0:
                LOG.info(f'Threshold: {threshold:.2f} - {batch_idx + 1} of '
                         f'{n_batches_per_thresh if n_batches_per_thresh is not None else "all"} batches done.')
    return float(np.mean(per_batch_fpr))


def calculate_fpr_minus_accepted(threshold: float, data_loader: DataLoader, model: torch.nn.Module, accepted_fpr: float,
                                 use_ground_truth: bool = False, n_batches_per_thresh: int = 50) -> float:
    """Returns the absolute difference of the mean FPR and an accepted FPR. Can be used to search threshold value."""
    mean_fpr = calculate_mean_false_positive_rate(threshold, data_loader, model, use_ground_truth, n_batches_per_thresh)
    return abs(mean_fpr - accepted_fpr)

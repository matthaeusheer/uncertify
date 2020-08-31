import math
import logging
import itertools

import scipy.stats
import torch
from torch.utils.data import DataLoader
import numpy as np

from uncertify.utils.custom_types import Tensor
from uncertify.evaluation.utils import threshold_batch_to_one_zero
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.utils.tensor_ops import print_scipy_stats_description

from typing import Generator, Dict, Tuple, Callable

LOG = logging.getLogger(__name__)


def l1_distance(t1: Tensor, t2: Tensor) -> Tensor:
    return torch.abs(t2 - t1)


def yield_reconstructed_batches(data_loader: DataLoader,
                                trained_model: torch.nn.Module,
                                max_batches: int = None,
                                residual_threshold: float = None,
                                residual_fn: Callable = l1_distance,
                                print_statistics: bool = False) -> Generator[Dict[str, Tensor], None, None]:
    """For some dataloader and a trained model, run the 'scan' tensors of the dataloader through the model
    and yield a tuple dicts of scan, reconstruction and (if present in dataloader) segmentation batches."""
    for batch in itertools.islice(data_loader, max_batches) if max_batches is not None else data_loader:
        scan_batch = batch['scan']
        reconstruction_batch = trained_model(scan_batch)[0]  # Maybe change model for not to do [0] here, not nice
        residual_batch = residual_fn(reconstruction_batch, scan_batch)
        thresholded_batch = normalize_to_0_1(residual_batch)
        if residual_threshold is not None:
            thresholded_batch = threshold_batch_to_one_zero(thresholded_batch, residual_threshold)
        output = {'scan': scan_batch, 'rec': reconstruction_batch, 'res': residual_batch, 'thresh': thresholded_batch}
        if 'seg' in batch.keys():
            # add segmentation if available
            output['seg'] = torch.where(batch['seg'] > 0, torch.ones_like(batch['seg']), torch.zeros_like(batch['seg']))
        if print_statistics:
            with torch.no_grad():
                for name, sub_batch in output.items():
                    description = scipy.stats.describe(sub_batch.detach().numpy().flatten())
                    print_scipy_stats_description(name=name, description_result=description)
        yield output


def yield_y_true_y_pred(data_loader: DataLoader,
                        trained_model: torch.nn.Module,
                        img_shape: Tuple[int, int],
                        max_n_batches: int = None,
                        residual_fn: Callable = l1_distance,
                        print_statistics: bool = False):
    """Yield flattened vectors over all (of max_n_batches, if not None) batches for y_true and y_pred.
    For Args: see yield_reconstruction_batches. Similar.
    Yields:
        tuple of y_true and y_pred, where
            y_true: a one-hot encoded vector where 1 stands for abnormal pixel and 0 stands for normal pixel
            y_pred: the residual vector (values ranging from 0 to 1)
    """
    batch_size = data_loader.batch_size
    data_loader = itertools.islice(data_loader, max_n_batches) if max_n_batches is not None else data_loader
    n_batches = len(data_loader) if max_n_batches is None else max_n_batches
    n_pixels_per_batch = math.prod(img_shape) * batch_size
    y_true = np.empty(n_batches * n_pixels_per_batch)
    y_pred = np.empty((n_batches * n_pixels_per_batch, 2))

    for batch_idx, batch in enumerate(yield_reconstructed_batches(data_loader, trained_model, max_n_batches,
                                                                  None, residual_fn, print_statistics)):
        with torch.no_grad():
            batch_y_true = batch['seg'].flatten().numpy()
            abnormal_pred = normalize_to_0_1(batch['res']).flatten().numpy()
            normal_pred = 1 - abnormal_pred
            stacked = np.vstack((normal_pred, abnormal_pred)).T
            y_true[batch_idx * n_pixels_per_batch: batch_idx * n_pixels_per_batch + n_pixels_per_batch] = batch_y_true
            y_pred[batch_idx * n_pixels_per_batch: batch_idx * n_pixels_per_batch + n_pixels_per_batch] = stacked
    return y_true, y_pred

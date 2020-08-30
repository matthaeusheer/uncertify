import logging
import itertools

import scipy.stats
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize

from uncertify.utils.custom_types import Tensor
from uncertify.evaluation.utils import threshold_batch_to_one_zero
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.utils.tensor_ops import print_scipy_stats_description

from typing import Generator, Dict, Callable

LOG = logging.getLogger(__name__)


def l1_distance(t1: Tensor, t2: Tensor) -> Tensor:
    return torch.abs(t2 - t1)


def yield_reconstructed_batches(dataloader: DataLoader,
                                trained_model: torch.nn.Module,
                                max_batches: int = None,
                                residual_threshold: float = 0.5,
                                residual_fn: Callable = l1_distance) -> Generator[Dict[str, Tensor], None, None]:
    """For some dataloader and a trained model, run the 'scan' tensors of the dataloader through the model
    and yield a tuple dicts of scan, reconstruction and (if present in dataloader) segmentation batches."""
    LOG.info(f'Running inference and yielding reconstructed & thresholded batches.')
    for batch in itertools.islice(dataloader, max_batches) if max_batches is not None else dataloader:
        scan_batch = batch['scan']
        reconstruction_batch = trained_model(scan_batch)[0]  # Maybe change model for not to do [0] here, not nice
        residual_batch = residual_fn(reconstruction_batch, scan_batch)
        thresholded_batch = threshold_batch_to_one_zero(normalize_to_0_1(residual_batch), residual_threshold)
        for sub_batch, name in [(scan_batch, 'scan'), (reconstruction_batch, 'reconstruction'),
                                (residual_batch, 'residual'), (thresholded_batch, 'thresholded')]:
            description = scipy.stats.describe(sub_batch.numpy().flatten())
            print_scipy_stats_description(name=name, description_result=description)
        output = {'scan': scan_batch, 'rec': reconstruction_batch, 'res': residual_batch, 'thresh': thresholded_batch}
        if 'seg' in batch.keys():
            # add segmentation if available
            output['seg'] = batch['seg']
        yield output

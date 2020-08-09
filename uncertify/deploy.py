import itertools

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize

from uncertify.utils.custom_types import Tensor

from typing import Generator, Dict, Callable


def l1_distance(t1: Tensor, t2: Tensor) -> Tensor:
    return torch.abs(t2 - t1)


def yield_reconstructed_batches(dataloader: DataLoader,
                                trained_model: torch.nn.Module,
                                max_batches: int = None,
                                residual_fn: Callable = l1_distance) -> Generator[Dict[str, Tensor], None, None]:
    """For some dataloader and a trained model, run the 'scan' tensors of the dataloader through the model
    and yield a tuple dicts of scan, reconstruction and (if present in dataloader) segmentation batches."""
    for batch in itertools.islice(dataloader, max_batches) if max_batches is not None else dataloader:
        scan_batch = batch['scan']
        reconstruction_batch = trained_model(scan_batch)[0]  # Maybe change model for not to do [0] here, not nice
        residual_batch = residual_fn(reconstruction_batch, scan_batch)
        for tensor in residual_batch:
            Normalize(mean=[0.0], std=[1.0], inplace=True)(tensor)
        if 'seg' in batch.keys():
            seg_batch = batch['seg']
            yield {'scan': scan_batch, 'rec': reconstruction_batch, 'res': residual_batch, 'seg': seg_batch}
        else:
            yield {'scan': scan_batch, 'rec': reconstruction_batch, 'res': residual_batch}


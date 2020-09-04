import itertools

import torch
from torch.utils.data import DataLoader


def get_n_masked_pixels(data_loader: DataLoader, max_n_batches: int = None) -> int:
    """Get the total number of masked pixels in the complete dataloader dataset."""
    n_masked_pixels = 0
    for batch in itertools.islice(data_loader, max_n_batches) if max_n_batches is not None else data_loader:
        n_masked_pixels += int(torch.sum(batch['mask']))
    return n_masked_pixels

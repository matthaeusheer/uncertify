import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import List, Tuple


def get_n_normal_abnormal_pixels(data_loader: DataLoader) -> Tuple[List[int], List[int], List[int]]:
    """Get the sample-wise numbers of normal and abnormal pixels.

    Note: Takes only pixels within brain mask into account and only samples which contain actual abnormal pixels.

    Returns:
        a tuple of two lists where the first list holds number of normal and second list number of abnormal pixels
        and the third list represents the number of pixels contained in the brain mask
    """
    normal_pixels = []
    abnormal_pixels = []
    total_masked_pixels = []
    for batch in tqdm(data_loader, total=len(data_loader), desc='Evaluating number of abnormal pixels'):
        assert 'seg' in batch.keys(), f'The batches in the dataloader need to provide "seg" entries.'
        for segmentation, mask in zip(batch['seg'], batch['mask']):
            masked_segmentation = segmentation[mask] > 0
            n_abnormal_pixels = torch.sum(masked_segmentation)
            if n_abnormal_pixels == 0:
                continue
            abnormal_pixels.append(int(n_abnormal_pixels))
            normal_pixels.append(int(masked_segmentation.numel() - n_abnormal_pixels))
            total_masked_pixels.append(int(masked_segmentation.numel()))
    return normal_pixels, abnormal_pixels, total_masked_pixels


def get_samples_without_lesions(data_loader: DataLoader) -> Tuple[int, int]:
    """Get a tuple representing (n_samples_without_any_lesions, total_number_of_samples)."""
    n_no_lesions = 0
    n_total = 0
    for batch in tqdm(data_loader, total=len(data_loader), desc=f'Determining samples without lesions'):
        for segmentation, mask in zip(batch['seg'], batch['mask']):
            masked_segmentation = segmentation[mask] > 0
            n_abnormal_pixels = torch.sum(masked_segmentation)
            if n_abnormal_pixels == 0:
                n_no_lesions += 1
            n_total += 1
    return n_no_lesions, n_total

import random
import torch

from uncertify.data.datasets import UncertifyDataset
from uncertify.data.utils import gauss_2d_tensor_image, gaussian

from typing import Tuple


class BrainGaussBlobDataset(UncertifyDataset):
    """
    This dataset applies artificial high intensity Gaussian blobs to an original wrapped dataset simulating lesions.
    """
    def __init__(self,
                 wrapped_dataset: UncertifyDataset,
                 min_pixels_to_consider: int = None,
                 max_sample_tries: int = None,
                 blob_weight: int = None,
                 std_min_max: Tuple[int, int] = None,
                 lesion_probability: float = None) -> None:
        super().__init__()
        self._wrapped_dataset = wrapped_dataset

        # Defining some default values if arguments not given
        self._min_pixels_to_consider = 100 if not min_pixels_to_consider else min_pixels_to_consider
        self._max_sample_tries = 100 if not max_sample_tries else max_sample_tries
        self._blob_weight = 1000 if not blob_weight else blob_weight
        self._std_min_max = (4, 10) if not std_min_max else std_min_max
        self._lesion_probability = 0.75 if not lesion_probability else lesion_probability

    @property
    def name(self) -> str:
        return '_'.join([self._wrapped_dataset.name, 'gauss_blobs'])

    def __len__(self) -> int:
        return len(self._wrapped_dataset)

    def __getitem__(self, idx) -> dict:
        item_dict = self._wrapped_dataset.__getitem__(idx)
        assert 'scan' in item_dict.keys(), f'Has no "scan" key - cannot wrap the dataset!'
        # Add a Gaussian blob only with certain probability
        seg = torch.zeros_like(item_dict['scan'])
        if random.random() > self._lesion_probability:
            item_dict['seg'] = seg
            return item_dict
        # Add the gaussian blob onto the scan within the masked region.
        n_pixels_within_mask = torch.sum(item_dict['mask'])
        if n_pixels_within_mask > self._min_pixels_to_consider:
            image = item_dict['scan'][0]
            mask = item_dict['mask'][0]
            height, width = image.shape
            assert height == width, f'So far does only work on images with equal size.'
            for _ in range(self._max_sample_tries):
                y_idx, x_idx = torch.randint(0, min(height, width), (2,))
                if mask[y_idx, x_idx]:
                    x_offset = -((width // 2) - x_idx)
                    y_offset = -((height // 2) - y_idx)
                    std = random.randint(*self._std_min_max)
                    gauss_blob = gauss_2d_tensor_image(grid_size=height, std=std, x_offset=x_offset, y_offset=y_offset)
                    gauss_filter = (self._blob_weight * gauss_blob)
                    threshold = gaussian(std, 0, std)
                    item_dict['scan'][0] += gauss_filter * mask
                    seg[0] = (gauss_filter > threshold) * mask
                    break
        item_dict['seg'] = seg
        return item_dict


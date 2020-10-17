from typing import List

import nibabel as nib
import numpy as np

from scripts.preprocess_brats import BACKGROUND_VALUE
from uncertify.data.preprocessing.histogram_matching.histogram_matching import MatchHistogramsTwoImages


def transform_images(images: np.ndarray) -> np.ndarray:
    """Do all manipulations to the raw numpy array like transposing, rotation etc."""
    images = np.transpose(images, axes=[2, 0, 1])  # makes dimensions to be [slice, width, height]?
    images = np.rot90(images, k=1, axes=(2, 1))  # rotates once in the (2, 1) plane, i.e. width-height-plane
    images = images[:, 27:227, 20:220]  # arbitrary numbers crop
    return images


def normalize_images(images: np.ndarray, masks: np.ndarray, normalization_method: str, background_value: float = None,
                     print_debug: bool = False) -> np.ndarray:
    """Performs normalization on all slices of  patient. Mean / max / min values are calculated per patient."""
    if print_debug:
        print(f'Performing normalization (zero mean, unit variance).')
    if normalization_method == 'standardize':
        images = (images - images[masks != 0].mean()) / images[masks != 0].std()
    elif normalization_method == 'rescale':
        images = (images - images[masks != 0].min()) / (images[masks != 0].max() - images[masks != 0].min())
    else:
        raise ValueError(f'Normalization method "{normalization_method}" unknown.')
    if background_value is not None:
        if print_debug:
            print(f'Set background to {BACKGROUND_VALUE}')
        images[masks == 0] = BACKGROUND_VALUE
    return images


def run_histogram_matching(input_images: np.ndarray, input_masks: np.ndarray,
                           reference_images: np.ndarray, reference_masks: np.ndarray,
                           print_debug: bool = False) -> np.ndarray:
    """Match histogram of (input_images, input_masks) against (reference_images, reference_masks)."""
    if print_debug:
        print(f'Performing histogram matching.')

    matched_img = MatchHistogramsTwoImages(reference_images, input_images, L=200, nbins=246, begval=0.05, finval=0.98,
                                           train_mask=reference_masks, test_mask=input_masks)
    return matched_img


def create_masks(images: np.ndarray) -> np.ndarray:
    """Get the masks for (already manipulated) sample slices."""
    mask = (images != 0).astype('int')
    return mask


def get_indices_to_keep(mask_file_path: str, exclude_empty_slices: bool = False) -> List[int]:
    """Returns a list of indices for a single patient mask slices where the mask is not empty."""
    mask_slices = nib.load(mask_file_path).get_fdata()
    if not exclude_empty_slices:
        return list(range(len(mask_slices)))
    else:
        non_empty_slice_indices = []
        for idx, img in enumerate(mask_slices):
            if np.count_nonzero(img) != 0:
                non_empty_slice_indices.append(idx)
        return non_empty_slice_indices
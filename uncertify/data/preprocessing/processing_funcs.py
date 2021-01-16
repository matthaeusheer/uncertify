from pathlib import Path

import nibabel as nib
import numpy as np
import cv2

from uncertify.data.preprocessing.histogram_matching.histogram_matching import MatchHistogramsTwoImages
from uncertify.data.preprocessing.preprocessing_config import CamCanConfig, BratsConfig, IBSRConfig, CANDIConfig, IXIConfig
from uncertify.data.preprocessing.preprocessing_config import BACKGROUND_VALUE

from typing import List, Union


def transform_images_brats(images: np.ndarray) -> np.ndarray:
    """Do all manipulations to the raw numpy array like transposing, rotation etc."""
    images = np.transpose(images, axes=[2, 0, 1])  # makes dimensions to be [slice, width, height]?
    images = np.rot90(images, k=1, axes=(2, 1))  # rotates once in the (2, 1) plane, i.e. width-height-plane
    images = images[:, 27:227, 20:220]  # arbitrary numbers crop
    return images


def transform_images_camcan(images: np.ndarray) -> np.ndarray:
    """Do all manipulations to the raw numpy array like transposing, rotation etc."""
    images = np.transpose(images, axes=[2, 0, 1])  # makes dimensions to be [slice, width, height]?
    images = np.rot90(images, k=3, axes=(1, 2))  # rotates once in the (2, 1) plane, i.e. width-height-plane
    images = pad_camcan_images(images)
    images = images[:, 27:227, 20:220]  # arbitrary numbers crop
    return images


def transform_images_ibsr(images: np.ndarray, is_mask: bool = False) -> np.ndarray:
    """Image transforms for IBSR data. Note that masks need other axis-rearrangement than scans and that the
    different pixel sizes require a resizing in the width direction after rotation."""
    # Remove last unused dimension
    if len(images.shape) == 4:
        images = images[:, :, :, 0]
    if is_mask:
        images = np.transpose(images, axes=(2, 0, 1))
    else:
        images = np.transpose(images, axes=(1, 0, 2))
    images = np.rot90(images, k=1, axes=(1, 2))
    n_axial_views, height, width = images.shape
    # Need to scale width-dimension
    scale_factor = 1.5
    width = int(width / scale_factor)
    axial_views = np.empty((n_axial_views, 200, 200))
    for axial_idx in range(n_axial_views):
        resized = cv2.resize(images[axial_idx], (width, height))
        center_width_idx = width // 2
        # Make image square
        square_img = resized[:, center_width_idx-128//2:center_width_idx+128//2]
        # Resize to (200, 200)
        axial_views[axial_idx, :, :] = cv2.resize(square_img, (200, 200))
    return axial_views


def transform_images_candi(images: np.ndarray, is_mask: bool = False) -> np.ndarray:
    """Image transforms for IBSR data. Note that masks need other axis-rearrangement than scans and that the
    different pixel sizes require a resizing in the width direction after rotation."""
    # Remove last unused dimension
    images = np.transpose(images, axes=(1, 0, 2))
    images = np.rot90(images, k=1, axes=(1, 2))
    n_axial_views, height, width = images.shape
    # Need to scale width-dimension
    scale_factor = 1.5
    width = int(width / scale_factor)
    axial_views = np.empty((n_axial_views, 200, 200))
    for axial_idx in range(n_axial_views):
        resized = cv2.resize(images[axial_idx], (width, height))
        center_width_idx = width // 2
        # Make image square
        square_img = resized[:, center_width_idx-128//2:center_width_idx+128//2]
        # Resize to (200, 200)
        axial_views[axial_idx, :, :] = cv2.resize(square_img, (200, 200))
    return axial_views


def pad_camcan_images(images: np.ndarray) -> np.ndarray:
    """Pad CamCAN images, s.t. they have the same shape as BraTS images (200x200).

    Shapes go from (189, 233, 197) -> (189, 240, 240).
    """
    # assert images.shape == (189, 197, 233), f'Not correct input shape {images.shape} for raw camcan patient.'
    return np.pad(images, [(0, 0), (4, 3), (21, 22)], mode='constant')


def normalize_images(images: np.ndarray, masks: np.ndarray, normalization_method: str,
                     background_value: float = BACKGROUND_VALUE, print_debug: bool = False) -> np.ndarray:
    """Performs normalization on all slices of  patient. Mean / max / min values are calculated per patient."""
    if print_debug:
        print(f'Performing normalization (zero mean, unit variance).')
    brain_pixels = images[masks != 0]
    if normalization_method == 'standardize':
        images = (images - brain_pixels.mean()) / brain_pixels.std()
    elif normalization_method == 'rescale':
        images = (images - brain_pixels.min()) / (brain_pixels.max() - brain_pixels.min())
    else:
        raise ValueError(f'Normalization method "{normalization_method}" unknown.')
    if background_value is not None:
        if print_debug:
            print(f'Set background to {background_value}')
        images[masks == 0] = background_value
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


def create_masks_camcan(images: np.ndarray) -> np.ndarray:
    """Get the masks for (already manipulated) sample slices."""
    mask = (images != 0).astype('int')
    return mask


def get_indices_to_keep(mask_file_path: Union[Path, np.ndarray], exclude_empty_slices: bool = False) -> List[int]:
    """Returns a list of indices for a single patient mask slices where the mask is not empty."""
    if isinstance(mask_file_path, Path):
        mask_slices = nib.load(mask_file_path).get_fdata()
    elif isinstance(mask_file_path, np.ndarray):
        mask_slices = mask_file_path
    else:
        raise TypeError(f'mask_file_path type not supported, must be string or np.ndarray, given {type(mask_file_path)}')
    if not exclude_empty_slices:
        return list(range(len(mask_slices)))
    else:
        non_empty_slice_indices = []
        for idx, img in enumerate(mask_slices):
            if np.count_nonzero(img) != 0:
                non_empty_slice_indices.append(idx)
        return non_empty_slice_indices


def create_hdf5_file_name(config: Union[BratsConfig, CamCanConfig, IBSRConfig, IXIConfig], train_or_val: str = 'train',
                          file_ending: str = '.hdf5') -> str:
    """Given the arguments passed into this script, create a meaningful filename for the created HDF5 file.."""
    assert train_or_val in {'train', 'val'}, f'Given train_or_val {train_or_val} not allowed.'
    is_brats = isinstance(config, BratsConfig)
    is_camcan = isinstance(config, CamCanConfig)
    is_ibsr = isinstance(config, IBSRConfig)
    is_candi = isinstance(config, CANDIConfig)
    is_ixi = isinstance(config, IXIConfig)

    name = config.dataset_name
    if is_camcan or is_ibsr or is_candi or is_ixi:
        name += f'_{train_or_val}'
    name += f'_{config.image_modality}'
    if config.do_histogram_matching:
        name += '_hm'
    if is_brats:
        if config.do_bias_correction:
            name += '_bc'
    if config.do_normalization:
        method = config.normalization_method
        name += '_std' if method == 'standardize' else '_scale' if method == 'rescale' else ''
        name += f'_bv{abs(config.background_value)}'
    if config.limit_to_n_samples is not None:
        name += f'_l{config.limit_to_n_samples}'
    if config.exclude_empty_slices:
        name += '_xe'
    name += file_ending
    return name

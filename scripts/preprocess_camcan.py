import os
import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm
import nibabel as nib
import h5py

import add_uncertify_to_path
from uncertify.data.preprocessing.preprocessing_config import CamCanConfig, preprocess_config_factory
from uncertify.data.preprocessing import camcan
from uncertify.data.preprocessing.processing_funcs import get_indices_to_keep
from uncertify.data.preprocessing.processing_funcs import run_histogram_matching
from uncertify.data.preprocessing.processing_funcs import transform_images_camcan
from uncertify.data.preprocessing.processing_funcs import create_masks
from uncertify.data.preprocessing.processing_funcs import normalize_images
from uncertify.data.preprocessing.processing_funcs import create_hdf5_file_name
from uncertify.utils.python_helpers import bool_to_str

from typing import Generator, Tuple


DEFAULT_CamCAN_ROOT_PATH    = Path('/scratch/maheer/datasets/raw/CamCAN')
REFERENCE_DIR_PATH          = Path('/scratch_net/samuylov/maheer/datasets/reference/')
HIST_REF_T1_PATH            = REFERENCE_DIR_PATH / 'sub-CC723395_T1w_unbiased.nii.gz'
HIST_REF_T1_MASK_PATH       = REFERENCE_DIR_PATH / 'sub-CC723395_T1w_brain_mask.nii.gz'
HIST_REF_T2_PATH            = REFERENCE_DIR_PATH / 'sub-CC723197_T2w_unbiased.nii.gz'
HIST_REF_T2_MASK_PATH       = REFERENCE_DIR_PATH / 'sub-CC723197_T2w_brain_mask.nii.gz'
HDF5_OUT_FOLDER             = Path('/scratch/maheer/datasets/processed/')
BACKGROUND_VALUE            = -3.5
VALID_MODALITIES            = ['t1', 't2']


def process_patients(config: CamCanConfig) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Main function to create an HDF5 dataset from a full CamCAN dataset."""
    img_paths = sorted(camcan.get_nii_sample_file_paths(config.dataset_root_path, config.image_modality))
    mask_paths = sorted(camcan.get_nii_mask_file_paths(config.dataset_root_path, config.image_modality))
    assert len(img_paths) == len(mask_paths), f'Not same amount of images and mask files. Abort.'

    # Check if limiting of patient is enabled
    if config.limit_to_n_samples is None:
        generator = zip(img_paths, mask_paths)
        n_total = len(img_paths)
    else:
        generator = zip(img_paths[:config.limit_to_n_samples], mask_paths[:config.limit_to_n_samples])
        n_total = config.limit_to_n_samples

    for img_path, mask_path in tqdm(list(generator), desc='Pre-processing CamCAN', total=n_total):
        images = nib.load(img_path).get_fdata()
        masks = nib.load(img_path).get_fdata()
        transformed_mask = create_masks(transform_images_camcan(images))  # needs to happen before messing with pixel values

        # Histogram matching against reference
        if config.do_histogram_matching:
            ref_img = nib.load(config.ref_paths[config.image_modality]['img']).get_fdata()
            ref_mask = nib.load(config.ref_paths[config.image_modality]['mask']).get_fdata()
            images = run_histogram_matching(images, masks, ref_img, ref_mask, config.print_debug)

        # Image transformation
        transformed_images = transform_images_camcan(images)

        # Normalization
        if config.do_normalization:
            transformed_images = normalize_images(transformed_images, transformed_mask, config.normalization_method,
                                                  config.background_value, config.print_debug)

        # Yield tuples of processed images and processed masks
        yield transformed_images, transformed_mask


def main_create_hdf5_dataset(config: CamCanConfig) -> None:
    """For a given config, run the whole pipeline to produce an HDF5 file output."""
    config.hdf5_out_folder_path.mkdir(parents=True, exist_ok=True)
    h5py_file = h5py.File(str(config.hdf5_out_folder_path / create_hdf5_file_name(config)), mode='w')

    create_new_dataset = True
    for images, masks in process_patients(config):
        # Possibly exclude empty slices
        keep_indices = get_indices_to_keep(masks, config.exclude_empty_slices)
        images = images[keep_indices]
        masks = masks[keep_indices]
        n_slices, height, width = images.shape
        slices = images.reshape(n_slices, width * height)
        masks = masks.reshape(n_slices, width * height)

        if create_new_dataset:
            h5py_file.create_dataset('scan', data=slices, maxshape=(None, width * height))
            h5py_file.create_dataset('mask', data=masks.astype('float'), maxshape=(None, width * height))
            create_new_dataset = False
        else:
            # Scan
            h5py_file['scan'].resize((h5py_file['scan'].shape[0] + len(slices)), axis=0)
            h5py_file['scan'][-len(slices):] = slices
            # Mask
            h5py_file['mask'].resize((h5py_file['mask'].shape[0] + len(masks)), axis=0)
            h5py_file['mask'][-len(masks):] = masks.astype('float')

    # Store processing metadata
    #for key, value in config.__dict__.items():
    #    h5py_file.attrs[key] = np.string_(value if type(value) is not bool else bool_to_str(value))
    h5py_file_path = h5py_file.filename
    h5py_file.close()
    print(f'Done creating h5py dataset. Output: {h5py_file_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process a CamCAN dataset and produce an HDF5 dataset.')
    parser.add_argument('--dataset-name',
                        default='default_dataset',
                        help='Name of the dataset used. E.g. "camcan".')

    parser.add_argument('--dataset-root-path',
                        type=Path,
                        default=DEFAULT_CamCAN_ROOT_PATH,
                        help='The root folder where the sub-folders for modalities exist.')

    parser.add_argument('-m',
                        '--modality',
                        nargs='+',
                        default='t2',
                        choices=VALID_MODALITIES,
                        help='Image modality to process.')

    parser.add_argument('-t',
                        '--limit-n-samples',
                        type=int,
                        default=None,
                        help='Handy for debugging. Limits the processed samples to this number.')

    parser.add_argument('-x',
                        '--exclude-empty-slices',
                        action='store_true',
                        help='If set, skips all empty slices when creating the hdf5 dataset. Note that in this case'
                             'a patient will possibly have less then the standard amount of slices and this'
                             'number may vary across patients.')

    parser.add_argument('-g',
                        '--no-histogram-matching',
                        action='store_true',
                        help='Perform histogram matching vs a reference sample.')

    parser.add_argument('-n',
                        '--no-normalization',
                        action='store_true',
                        help='Skips normalization for the masked pixels.')

    parser.add_argument('--normalization-method',
                        type=str,
                        default='rescale',
                        choices=['rescale', 'standardize'],
                        help='How to normalize.')

    parser.add_argument('--hdf5-out-dir-path',
                        type=Path,
                        default=HDF5_OUT_FOLDER,
                        help='Location to store the final HDF5 output file.')

    parser.add_argument('-d',
                        '--print-debug',
                        action='store_true',
                        help='If set, enables debug print information on stdout.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    run_config = preprocess_config_factory(parse_args(), dataset_type='camcan')
    pprint(run_config.__dict__)
    main_create_hdf5_dataset(run_config)


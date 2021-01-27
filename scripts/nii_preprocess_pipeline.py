import os
import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
from tqdm import tqdm
import nibabel as nib
import h5py
from sklearn.model_selection import train_test_split

import add_uncertify_to_path
import uncertify
from uncertify.data.preprocessing.camcan import get_camcan_nii_sample_file_paths, get_camcan_nii_mask_file_paths
from uncertify.data.preprocessing.ibsr import get_isbr2_nii_file_paths, get_isbr2_sample_dir_paths
from uncertify.data.preprocessing.candi import get_candi_nii_file_paths, get_candi_sample_dir_paths
from uncertify.data.preprocessing.preprocessing_config import CamCanConfig, IBSRConfig, PreprocessConfig, CANDIConfig, \
    IXIConfig
from uncertify.data.preprocessing.preprocessing_config import preprocess_config_factory
from uncertify.data.preprocessing.processing_funcs import get_indices_to_keep
from uncertify.data.preprocessing.processing_funcs import run_histogram_matching
from uncertify.data.preprocessing.processing_funcs import transform_images_camcan, transform_images_ibsr, \
    transform_images_candi
from uncertify.data.preprocessing.processing_funcs import create_masks_camcan
from uncertify.data.preprocessing.processing_funcs import normalize_images
from uncertify.data.preprocessing.processing_funcs import create_hdf5_file_name
from uncertify.common import DATA_DIR_PATH

from typing import Generator, Tuple, List, Union

DEFAULT_CamCAN_ROOT_PATH = DATA_DIR_PATH / 'raw' / 'CamCAN'
DEFAULT_IBSR_ROOT_PATH = Path('/mnt/2TB_internal_HD/datasets/raw/IBSR/IBSR_V2.0_nifti_stripped/IBSR_nifti_stripped')

REFERENCE_DIR_PATH = DATA_DIR_PATH / 'reference' / 'CamCAN'
HIST_REF_T1_PATH = REFERENCE_DIR_PATH / 'sub-CC420202_T1w_unbiased.nii.gz'
HIST_REF_T1_MASK_PATH = REFERENCE_DIR_PATH / 'sub-CC420202_T1w_brain_mask.nii.gz'
HIST_REF_T2_PATH = REFERENCE_DIR_PATH / 'sub-CC723197_T2w_unbiased.nii.gz'
HIST_REF_T2_MASK_PATH = REFERENCE_DIR_PATH / 'sub-CC723197_T2w_brain_mask.nii.gz'
HDF5_OUT_FOLDER = DATA_DIR_PATH / 'processed'
VALID_MODALITIES = ['t1', 't2']  # For CamCAN
DEFAULT_VAL_SET_FRACTION = 0.1
TRAIN_VAL_SPLIT_RANDOM_SEED = 42
DEFAULT_BACKGROUND_VALUE = 0.0


class PreProcessingPipeline:
    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config

    def process_patients(self, img_mask_tuple_list: list,
                         train_or_val: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Given a list of image/mask tuples, yield processed imaeg/mask iamges as numpy arrays."""
        raise NotImplementedError

    def create_hdf5_dataset(self) -> None:
        """For a given config, run the whole pipeline to produce an HDF5 file output."""
        img_mask_tuple_list = self.get_nii_file_paths_img_mask_tuples()
        train_img_mask_tuple_list, val_img_mask_tuple_list = train_test_split(img_mask_tuple_list,
                                                                              test_size=DEFAULT_VAL_SET_FRACTION,
                                                                              random_state=TRAIN_VAL_SPLIT_RANDOM_SEED)
        set_paths_dict = {
            'train': train_img_mask_tuple_list,
            'val': val_img_mask_tuple_list
        }
        self._config.hdf5_out_folder_path.mkdir(parents=True, exist_ok=True)

        for train_or_val, img_mask_tuple_list in set_paths_dict.items():
            h5py_file = h5py.File(str(self._config.hdf5_out_folder_path / create_hdf5_file_name(self._config,
                                                                                                train_or_val)),
                                  mode='w')

            create_new_dataset = True
            for images, masks in self.process_patients(img_mask_tuple_list, train_or_val):
                # Possibly exclude empty slices
                keep_indices = get_indices_to_keep(masks, self._config.exclude_empty_slices)
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
            # for key, value in self._config.__dict__.items():
            #    h5py_file.attrs[key] = np.string_(value if type(value) is not bool else bool_to_str(value))
            h5py_file_path = h5py_file.filename
            h5py_file.close()
            print(f'Done creating h5py dataset. Output: {h5py_file_path}')

    def get_nii_file_paths_img_mask_tuples(self) -> List[Tuple[Path, Path]]:
        """Get a list of (img_path, mask_path) tuples. One tuple per patient."""
        if isinstance(self._config, IBSRConfig):
            sample_dir_paths = get_isbr2_sample_dir_paths(self._config.dataset_root_path)
            img_paths = sorted(get_isbr2_nii_file_paths(sample_dir_paths, 'ana'))
            mask_paths = sorted(get_isbr2_nii_file_paths(sample_dir_paths, 'ana_brainmask'))
        elif isinstance(self._config, CANDIConfig):
            sample_dir_paths = get_candi_sample_dir_paths(self._config.dataset_root_path)
            img_paths = sorted(get_candi_nii_file_paths(sample_dir_paths, '_procimg'))
            mask_paths = sorted(get_candi_nii_file_paths(sample_dir_paths, '.seg'))
        elif isinstance(self._config, CamCanConfig):
            # No masks available out of the box, img and mask paths will be the same here (both images actually)
            img_paths = sorted(
                get_camcan_nii_sample_file_paths(self._config.dataset_root_path, self._config.image_modality))
            mask_paths = sorted(
                get_camcan_nii_mask_file_paths(self._config.dataset_root_path, self._config.image_modality))
        elif isinstance(self._config, IXIConfig):
            img_paths = sorted(
                get_camcan_nii_sample_file_paths(self._config.dataset_root_path, self._config.image_modality,
                                                 keyword=None))
            mask_paths = img_paths
        else:
            raise TypeError(f'Config type {self._config.__class__} not supported.')
        assert len(img_paths) == len(mask_paths), f'Not same amount of images and mask files. Abort.'

        # Check if limiting of patient is disabled / enabled
        if self._config.limit_to_n_samples is None:
            generator = zip(img_paths, mask_paths)
        else:
            generator = zip(img_paths[:self._config.limit_to_n_samples], mask_paths[:self._config.limit_to_n_samples])
        img_mask_tuple_list = list(generator)
        return img_mask_tuple_list


class PreProcessingPipelineCamCanIXI(PreProcessingPipeline):
    def __init__(self, config: Union[CamCanConfig, IXIConfig]) -> None:
        super().__init__(config)

    def process_patients(self, img_mask_tuple_list: list,
                         train_or_val: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_patients = len(img_mask_tuple_list)
        for img_path, mask_path in tqdm(list(img_mask_tuple_list),
                                        desc=f'Pre-processing CamCAN/IXI {train_or_val}', total=n_patients):
            images = nib.load(img_path).get_fdata()
            masks = nib.load(img_path).get_fdata()
            # creating transformed mask needs to happen before messing with pixel values (normalization etc.)
            transformed_mask = create_masks_camcan(transform_images_camcan(images))

            # Histogram matching against reference
            if self._config.do_histogram_matching:
                try:
                    ref_img = nib.load(self._config.ref_paths[self._config.image_modality]['img']).get_fdata()
                except FileNotFoundError:
                    print(f'Did not find the reference slices to match histograms against.')
                    raise
                try:
                    ref_mask = nib.load(self._config.ref_paths[self._config.image_modality]['mask']).get_fdata()
                except FileNotFoundError:
                    print(f'Did not find the reference slices masks to match histograms against.')
                    raise
                images = run_histogram_matching(images, masks, ref_img, ref_mask, self._config.print_debug)

            # Image transformation
            transformed_images = transform_images_camcan(images)

            # Normalization
            if self._config.do_normalization:
                transformed_images = normalize_images(transformed_images, transformed_mask,
                                                      self._config.normalization_method,
                                                      self._config.background_value, self._config.print_debug)

            # Yield tuples of processed images and processed masks
            yield transformed_images, transformed_mask


class PreProcessingPipelineIBSRCANDI(PreProcessingPipeline):

    def process_patients(self, img_mask_tuple_list: list,
                         train_or_val: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_patients = len(img_mask_tuple_list)
        for img_path, mask_path in tqdm(list(img_mask_tuple_list),
                                        desc=f'Pre-processing IBSR/CANDI {train_or_val}', total=n_patients):
            images = nib.load(img_path).get_fdata()
            masks = nib.load(mask_path).get_fdata()
            if isinstance(self._config, IBSRConfig):
                transformed_images = transform_images_ibsr(images, is_mask=False)
                transformed_masks = transform_images_ibsr(masks, is_mask=True)
            elif isinstance(self._config, CANDIConfig):
                transformed_images = transform_images_candi(images)
                transformed_masks = transform_images_candi(masks)
            else:
                raise TypeError(f'Wrong config type.')

            # Histogram matching against reference
            if self._config.do_histogram_matching:
                raise NotImplementedError(f'Histogram matching not implemented for IBSR/CANDI.')
                try:
                    ref_img = nib.load(self._config.ref_paths[self._config.image_modality]['img']).get_fdata()
                except FileNotFoundError:
                    print(f'Did not find the reference slices to match histograms against.')
                    raise
                try:
                    ref_mask = nib.load(self._config.ref_paths[self._config.image_modality]['mask']).get_fdata()
                except FileNotFoundError:
                    print(f'Did not find the reference slices masks to match histograms against.')
                    raise
                images = run_histogram_matching(images, masks, ref_img, ref_mask, self._config.print_debug)

            # Normalization
            if self._config.do_normalization:
                transformed_images = normalize_images(transformed_images, transformed_masks,
                                                      self._config.normalization_method,
                                                      self._config.background_value, self._config.print_debug)

            # Yield tuples of processed images and processed masks
            yield transformed_images, transformed_masks


def preprocess_pipeline_factory(config: PreprocessConfig) -> PreProcessingPipeline:
    if isinstance(config, CamCanConfig) or isinstance(config, IXIConfig):
        return PreProcessingPipelineCamCanIXI(config)
    elif isinstance(config, IBSRConfig) or isinstance(config, CANDIConfig):
        return PreProcessingPipelineIBSRCANDI(config)
    else:
        raise TypeError(f'Preprocessing config not supported.')


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process a CamCAN dataset and produce an HDF5 dataset.')
    parser.add_argument('--dataset-name',
                        default='camcan',
                        choices=['camcan', 'ibsr', 'candi', 'ixi'],
                        help='Name / type of the dataset used.')

    parser.add_argument('--dataset-root-path',
                        type=Path,
                        default=DEFAULT_CamCAN_ROOT_PATH,
                        help='The root folder where the sub-folders for modalities/samples exist '
                             '(depending on dataset-name/type).')

    parser.add_argument('-m',
                        '--modality',
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

    parser.add_argument('--val-fraction',
                        type=float,
                        default=DEFAULT_VAL_SET_FRACTION,
                        help='Fraction of data to be used for validation (other be used for training).')
    parser.add_argument('--background-value',
                        type=float,
                        default=DEFAULT_BACKGROUND_VALUE,
                        help='Value to be set outside the brain mask.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    ref_paths = {'t1': {'img': HIST_REF_T1_PATH,
                        'mask': HIST_REF_T1_MASK_PATH},
                 't2': {'img': HIST_REF_T2_PATH,
                        'mask': HIST_REF_T2_MASK_PATH}
                 }
    args = parse_args()
    run_config = preprocess_config_factory(args, ref_paths, dataset_type=args.dataset_name)
    print(f'Pre-processing config: {run_config.dataset_name}')
    pprint(run_config.__dict__)

    preprocessing_pipeline = preprocess_pipeline_factory(run_config)
    preprocessing_pipeline.create_hdf5_dataset()

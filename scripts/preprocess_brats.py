import os
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
import subprocess
from pprint import pprint
from dataclasses import dataclass

import numpy as np
import h5py
import nibabel as nib
import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.utils.python_helpers import bool_to_str
from uncertify.data.preprocessing.histogram_matching.histogram_matching import MatchHistogramsTwoImages

from typing import Tuple, List, Dict

VALID_MODALITIES = ['t1', 't2', 'flair', 't1ce', 'seg']
BACKGROUND_VALUE = -3.5
DEFAULT_BRATS_GLOB_PATH = f'/scratch/maheer/datasets/raw/BraTS2017/training/*/*'
DEFAULT_BRATS_ROOT_PATH = Path('/scratch/maheer/datasets/raw/BraTS2017/training')
HIST_REF_T1_PATH = Path('/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t1_unbiased.nii.gz')
HIST_REF_T1_MASK_PATH = Path(
    '/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t1_unbiased_mask.nii.gz')
HIST_REF_T2_PATH = Path('/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t2_unbiased.nii.gz')
HIST_REF_T2_MASK_PATH = Path(
    '/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t2_mask_unbiased.nii.gz')
HDF5_OUT_FOLDER = Path('/scratch/maheer/datasets/processed/')
DEFAULT_DATASET_NAME = 'BraTS17'
N4_EXECUTABLE_PATH = Path('/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4')


@dataclass
class PreprocessConfig:
    """A single config which gets populated via the parsed command line arguments used throughout a processing run."""
    dataset_name: str
    dataset_root_path: Path
    do_pre_processing: bool
    do_create_dataset: bool
    modalities: Dict[str, bool]
    limit_to_n_samples: int
    exclude_empty_slices: bool
    do_bias_correction: bool
    force_bias_correction: bool
    do_histogram_matching: bool
    ref_paths: Dict[str, Dict[str, Path]]
    do_normalization: bool
    normalization_method: str
    shuffle_pre_processing: bool
    background_value: float
    hdf5_out_folder_path: Path
    n4_executable_path: Path
    store_pre_processing_output: bool
    print_debug: bool


def preprocess_config_factory(args: argparse.Namespace) -> PreprocessConfig:
    """Factory method to create a pre-processing config based on the parsed command line arguments."""
    config = PreprocessConfig(
        dataset_name=args.dataset_name,
        dataset_root_path=args.dataset_root_path,
        do_pre_processing=args.pre_process,
        do_create_dataset=args.create_dataset,
        modalities={modality: modality in args.modalities for modality in VALID_MODALITIES},
        limit_to_n_samples=args.limit_n_samples,
        exclude_empty_slices=args.exclude_empty_slices,
        do_bias_correction=not args.no_bias_correction,
        force_bias_correction=args.force_bias_correction,
        do_histogram_matching=not args.no_histogram_matching,
        ref_paths={'t1': {'img': HIST_REF_T1_PATH,
                          'mask': HIST_REF_T1_MASK_PATH},
                   't2': {'img': HIST_REF_T2_PATH,
                          'mask': HIST_REF_T2_MASK_PATH}
                   },
        do_normalization=not args.no_normalization,
        normalization_method=args.normalization_method,
        shuffle_pre_processing=args.shuffle_pre_processing,
        background_value=BACKGROUND_VALUE,
        hdf5_out_folder_path=args.hdf5_out_dir_path,
        n4_executable_path=N4_EXECUTABLE_PATH,
        store_pre_processing_output=not args.no_output,
        print_debug=args.print_debug
    )
    return config


def validate_preprocess_config(config: PreprocessConfig) -> PreprocessConfig:
    """Validates that the config can produce a valid pre-processing run and return the valid config."""
    return config


def run_preprocessing(config: PreprocessConfig) -> None:
    """Main pre-processing function which performs the pre-processing pipeline for all modalities and patients.

    Note:   This method creates additional output files in the original BraTS data folder. That is, for
            example, next to Brats17_TCIA_117_1_t1.nii.gz it will create Brats17_TCIA_117_1_t1_processed.nii.gz
            Those files are later read by the create_dataset function to create the HDF5 dataset. Original files are
            not altered.
    """
    # TODO: Potentially all sub-folders?
    sample_dir_paths = glob.glob(str(config.dataset_root_path / '*/*'))
    sample_dir_paths = [Path(path) for path in sample_dir_paths if 'Brats17_2013' not in os.path.split(path)[-1]]
    if config.shuffle_pre_processing:
        np.random.shuffle(sample_dir_paths)
    if config.limit_to_n_samples is not None:
        sample_dir_paths = sample_dir_paths[:config.limit_to_n_samples]

    for sample_dir_path in tqdm(sample_dir_paths, desc='pre-processing patients'):
        if config.print_debug:
            print(f'\n ------ Processing {sample_dir_path} ------')
        for modality in [mod for mod in config.modalities if config.modalities[mod] is True]:
            process_modality(modality=modality, sample_dir_path=sample_dir_path, config=config)


def process_modality(modality: str, sample_dir_path: Path, config: PreprocessConfig) -> None:
    """Process the images for a modality and store processed nii files at same location like originals.

    Without un-biasing output file names will be <sample_dir_name>_<modality>.nii.gz.
    With un-biasing output file names will be <sample_dir_name>_<modality>_unbiased.nii.gz
    """
    file_name_orig = create_nii_file_name(sample_dir_path.name, modality,
                                          is_unbiased=False, is_mask=False, is_processed=False)
    file_name_orig_unbiased = create_nii_file_name(sample_dir_path.name, modality,
                                                   is_unbiased=True, is_mask=False, is_processed=False)

    # Bias correction
    if config.do_bias_correction and modality != 'seg':
        run_bias_correction(sample_dir_path, file_name_orig, file_name_orig_unbiased, modality, config)

    # Set file name we use to do further processing with
    file_name = file_name_orig_unbiased if config.do_bias_correction and not modality == 'seg' else file_name_orig
    nii_file_path = sample_dir_path / file_name
    if config.print_debug:
        print(f'Processing {modality} slices: {nii_file_path}')

    # Load actual NII data file
    slices: np.array = nib.load(nii_file_path).get_fdata()
    original_mask = create_masks(slices)
    transformed_mask = create_masks(transform_image(slices))

    # Histogram matching
    if config.do_histogram_matching and modality != 'seg':
        slices = run_histogram_matching(slices,
                                        original_mask,
                                        str(config.ref_paths[modality]['img']),
                                        str(config.ref_paths[modality]['mask']),
                                        config.print_debug)

    # Image transformation (transpose, rotate, crop, etc.)
    transformed_slices = transform_image(slices)

    # Normalization
    if config.do_normalization and modality != 'seg':
        transformed_slices = normalize_image(transformed_slices, transformed_mask, config.normalization_method,
                                             config.background_value, config.print_debug)
    # Crush segmentation labels [1, 4] all to 1
    if modality == 'seg':
        transformed_slices = np.where(transformed_slices > 0, np.ones(transformed_slices.shape),
                                      np.zeros(transformed_slices.shape))

    # Storing output
    if config.store_pre_processing_output:
        store_pre_processing_output(transformed_slices, transformed_mask, sample_dir_path, modality,
                                    config.do_bias_correction, config.print_debug)


def transform_image(slices: np.ndarray) -> np.ndarray:
    """Do all manipulations to the raw numpy array like transposing, rotation etc."""
    slices = np.transpose(slices, axes=[2, 0, 1])  # makes dimensions to be [slice, width, height]?
    slices = np.rot90(slices, k=1, axes=(2, 1))  # rotates once in the (2, 1) plane, i.e. width-height-plane
    slices = slices[:, 27:227, 20:220]  # arbitrary numbers crop
    return slices


def normalize_image(slices: np.ndarray, masks: np.ndarray, normalization_method: str, background_value: float = None,
                    print_debug: bool = False) -> np.ndarray:
    """Performs normalization on all slices of  patient. Mean / max / min values are calculated per patient."""
    if print_debug:
        print(f'Performing normalization (zero mean, unit variance).')
    if normalization_method == 'standardize':
        slices = (slices - slices[masks != 0].mean()) / slices[masks != 0].std()
    elif normalization_method == 'rescale':
        slices = (slices - slices[masks != 0].min()) / (slices[masks != 0].max() - slices[masks != 0].min())
    else:
        raise ValueError(f'Normalization method "{normalization_method}" unknown.')
    if background_value is not None:
        if print_debug:
            print(f'Set background to {BACKGROUND_VALUE}')
        slices[masks == 0] = BACKGROUND_VALUE
    return slices


def run_bias_correction(dir_path: Path, in_file_name: str, out_file_name: str, modality: str, config: PreprocessConfig) -> None:
    """Depending on setting regarding forcing the bias correction, run it for some modality."""
    if not (dir_path / out_file_name).exists():
        run_bias_correction_single_file(dir_path, in_file_name, out_file_name, modality, config.print_debug)
    else:
        if config.force_bias_correction:
            run_bias_correction_single_file(dir_path, in_file_name, out_file_name, modality, config.print_debug)
        else:
            if config.print_debug:
                print(f'Using existing unbiased file (no bias correction force): {out_file_name}')


def run_bias_correction_single_file(dir_path: Path, in_file_name: str, out_file_name: str,
                                    modality: str, print_debug: bool = False) -> None:
    """Run bias correction on a single file.

    Creates a new file in the same location with '<name>_unbiased.nii.gz'.
    Creates a new file in the same location for mask '<name>_mask_unbiased.nii.gz'"""
    command = ' '.join([str(N4_EXECUTABLE_PATH), in_file_name, out_file_name])
    if print_debug:
        print(f'Running bias correction: {command}')
    subprocess.run(command, shell=True, cwd=dir_path)
    # Now store mask because we might need it for reference histogram matching
    slices = nib.load(dir_path / out_file_name).get_fdata()
    mask = create_masks(slices)
    out_mask_name = create_nii_file_name(dir_path.name, modality, is_mask=True, is_unbiased=True, is_processed=False)
    nif_masks = nib.Nifti1Image(mask, np.eye(4))
    nib.save(nif_masks, dir_path / out_mask_name)
    if print_debug:
        print(f'Saved unbiased mask: {out_mask_name}')


def run_histogram_matching(img: np.ndarray, orig_mask: np.ndarray, ref_img_path: str, ref_mask_path: str,
                           print_debug: bool = False) -> np.ndarray:
    if print_debug:
        print(f'Performing histogram matching with {ref_img_path}')
    ref_img = nib.load(ref_img_path).get_fdata()
    ref_mask = nib.load(ref_mask_path).get_fdata()
    matched_img = MatchHistogramsTwoImages(ref_img, img, L=200, nbins=246, begval=0.05, finval=0.98,
                                           train_mask=ref_mask,
                                           test_mask=orig_mask)
    return matched_img


def store_pre_processing_output(slices: np.ndarray, masks: np.ndarray, sample_dir_path: Path,
                                modality: str, is_unbiased: bool, print_debug: bool = False) -> None:
    """Store the processed image slices and mask (both are transformed!) to the same folder as input."""
    # First the nif image for the slices
    nif_slices = nib.Nifti1Image(slices, np.eye(4))
    out_file_name = create_nii_file_name(sample_dir_path.name, modality,
                                         is_mask=False, is_unbiased=is_unbiased, is_processed=True)
    nib.save(nif_slices, sample_dir_path / out_file_name)
    if print_debug:
        print(f'Saved {modality} slices sample to {sample_dir_path / out_file_name}.')

    # Then the brain mask
    if modality != 'seg':  # For the segmentation slices it doesn't make sense to store a mask
        nif_mask = nib.Nifti1Image(masks, np.eye(4))
        out_file_name_mask = create_nii_file_name(sample_dir_path.name, modality,
                                                  is_mask=True, is_unbiased=False, is_processed=True)
        nib.save(nif_mask, sample_dir_path / out_file_name_mask)
        print(f'Saved {modality} mask sample to {sample_dir_path / out_file_name_mask}.')


def create_masks(slices: np.ndarray) -> np.ndarray:
    """Get the masks for (already manipulated) sample slices."""
    mask = (slices != 0).astype('int')
    return mask


def create_dataset(config: PreprocessConfig) -> None:
    """After pre-processing a whole dataset (writing processed images to disk), this function creates an HDF5
    dataset using h5py."""
    config.hdf5_out_folder_path.mkdir(parents=True, exist_ok=True)
    h5py_file = h5py.File(str(config.hdf5_out_folder_path / create_dataset_name(config)), 'w')
    # +1 for mask, new_col whether a dataset has already been created
    new_col = sum([config.modalities['t1'], config.modalities['t2'], config.modalities['seg']]) + 1

    sample_dir_paths = glob.glob(str(config.dataset_root_path / '*/*'))
    sample_dir_paths = [Path(item) for item in sample_dir_paths if 'Brats17_2013' not in os.path.split(item)[-1]]

    n_processed = 0
    for sample_dir_path in tqdm(sample_dir_paths, desc='Creating HDF5 dataset',
                                total=config.limit_to_n_samples if
                                config.limit_to_n_samples is not None else len(sample_dir_paths)):
        if config.print_debug:
            print(f'\nProcessing {sample_dir_path}')

        image_modality = get_image_modality(config)
        if config.do_histogram_matching:
            if sample_dir_path.name in config.ref_paths[image_modality]['img'].name:
                n_processed += 1  # Could encounter a sample which we have not processed yet. Dataset will be size - 1
                continue # Exclude sample we match against!
        for modality in [mod for mod in config.modalities if config.modalities[mod] is True]:

            # Add mask which is always done
            nii_mask_name = create_nii_file_name(sample_dir_path.name, image_modality,
                                                 is_mask=True, is_unbiased=False, is_processed=True)
            if config.print_debug:
                print(f'Loading mask {nii_mask_name}')
            masks_path = sample_dir_path / nii_mask_name
            keep_indices = get_indices_to_keep(masks_path, config.exclude_empty_slices)

            if modality != 'seg':  # Does not make sense for segmentation map to add brain mask
                h5py_file, new_col = add_dataset_to_h5py(masks_path, h5py_file, 'mask', new_col, keep_indices)

            # Add other modality (segmentation and possibly another one)
            dataset_key = 'seg' if modality == 'seg' else 'scan'
            nii_slices_name = create_nii_file_name(sample_dir_path.name, modality,
                                                   is_mask=False, is_unbiased=config.do_bias_correction,
                                                   is_processed=True)
            if config.print_debug:
                print(f'Loading slices {nii_slices_name}')
            nii_slices_path = sample_dir_path / nii_slices_name
            h5py_file, new_col = add_dataset_to_h5py(nii_slices_path, h5py_file, dataset_key, new_col, keep_indices)

        n_processed += 1
        if n_processed == config.limit_to_n_samples:
            break

    # Store processing metadata
    for key, value in config.__dict__.items():
        h5py_file.attrs[key] = np.string_(value if type(value) is not bool else bool_to_str(value))
    h5py_file_path = h5py_file.filename
    h5py_file.close()
    print(f'Done creating h5py dataset. Output: {h5py_file_path}')


def get_image_modality(config: PreprocessConfig) -> str:
    """Only one image modality per dataset is allowed. Get this one."""
    modalities_dict = config.modalities
    # TODO: This is rather dangerous.
    image_modality = [key for key in modalities_dict.keys() if key != 'seg' and modalities_dict[key] is True][0]
    return image_modality


def add_dataset_to_h5py(file_path: str, h5py_file: h5py.File, dataset_key: str, new_col: int,
                        keep_indices: List[int] = None) -> Tuple[h5py.File, int]:
    """Adds a new sample (full patient) to a h5py file. New col determines if the sample is added to an existing
    dataset or a new one is created within the h5py file."""
    slices = nib.load(file_path).get_fdata()[keep_indices]
    n_slices, height, width = slices.shape

    slices = slices.reshape(n_slices, width * height)
    if new_col:
        h5py_file.create_dataset(dataset_key, data=slices, maxshape=(None, 200 * 200))
        new_col = new_col - 1
    else:
        h5py_file[dataset_key].resize((h5py_file[dataset_key].shape[0] + len(slices)), axis=0)
        h5py_file[dataset_key][-len(slices):] = slices
    return h5py_file, new_col


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


def create_nii_file_name(sample_dir_name: str, modality: str, is_mask: bool, is_unbiased: bool,
                         is_processed: bool) -> str:
    """File name which will be created for pre-processed output, e.g. Brats17_TCIA_131_1_t1_unbiased_processed.nii.gz"""
    file_name = f'{sample_dir_name}' \
                f'{f"_{modality}" if not (modality == "seg" and is_mask) else ""}' \
                f'{"_unbiased" if (is_unbiased and modality != "seg") else ""}' \
                f'{"_mask" if is_mask else ""}' \
                f'{"_processed" if is_processed else ""}.nii.gz'
    return file_name


def create_dataset_name(config: PreprocessConfig, file_ending: str = '.hdf5') -> str:
    """Given the arguments passed into this script, create a meaningful filename for the created HDF5 file.."""
    name = config.dataset_name
    for modality in [mod for mod in config.modalities.keys() if mod not in ['seg'] and config.modalities[mod] is True]:
        name += f'_{modality}'
    if config.do_histogram_matching:
        name += '_hm'
    if config.do_bias_correction:
        name += '_bc'
    if config.do_normalization:
        method = config.normalization_method
        name += '_std' if method == 'standardize' else '_scale' if method == 'rescale' else ''
    if config.limit_to_n_samples is not None:
        name += f'_l{config.limit_to_n_samples}'
    if config.exclude_empty_slices:
        name += '_xe'
    name += file_ending
    return name


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process a BraTS dataset and / or create a '
                                                 'HDF5 dataset out of it.')
    parser.add_argument('--dataset-name',
                        help='Name of the dataset used. E.g. "brats17".')

    parser.add_argument('--dataset-root-path',
                        type=Path,
                        default=DEFAULT_BRATS_ROOT_PATH,
                        help='The root folder where the nii files (possibly in subfolders) exist.')

    parser.add_argument('-p',
                        '--pre-process',
                        action='store_true',
                        help='If set, performs pre-processing of the whole dataset. '
                             'Output for the respective modalities will be stored at the same folder like the '
                             'original sample with "processed" prepended before the .nii.gz ending.')

    parser.add_argument('-c',
                        '--create-dataset',
                        action='store_true',
                        help='If set, creates an HDF5 dataset out of the processed dataset (see -p above).')

    parser.add_argument('-m',
                        '--modalities',
                        nargs='+',
                        default=['t1', 'seg'],
                        choices=VALID_MODALITIES,
                        help='List of modalities to process.')

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

    parser.add_argument('-b',
                        '--no-bias-correction',
                        action='store_true',
                        help='Perform bias correction during/before pre-processing. If unbiased version '
                             'already present use this one unless --force-bias-correction is set.')

    parser.add_argument('-f',
                        '--force-bias-correction',
                        action='store_true',
                        help='If bias correction enabled (using -b) and a bias corrected version already exists, '
                             'i.e. a "<file_name>_unbiased_nii.gz", only perform bias correction again when this '
                             'flag is enabled, otherwise take the one present already.')

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

    parser.add_argument('-s',
                        '--no-output',
                        action='store_true',
                        help='Skip storing output into "<name>_processed.nii.gz" files '
                             'at the same location during pre-processing step.')

    parser.add_argument('-l',
                        '--shuffle-pre-processing',
                        action='store_true',
                        help='Whether to shuffle the samples during pre-processing.')

    parser.add_argument('--hdf5-out-dir-path',
                        type=Path,
                        default=HDF5_OUT_FOLDER,
                        help='Location to store the final HDF5 output file.')

    parser.add_argument('-d',
                        '--print-debug',
                        action='store_true',
                        help='If set, enables debug print information on stdout.')
    args = parser.parse_args()
    if 't1' in args.modalities and 't2' in args.modalities:
        raise ValueError(f'Multiple imaging modalities in one dataset not supported currently.')
    if not any([args.pre_process, args.create_dataset]):
        raise ValueError(f'Choose either -p (pre-process), -c (create-dataset) or both. '
                         f'You have not chosen at least one.')
    return args


def main(args: argparse.Namespace) -> None:
    config: PreprocessConfig = preprocess_config_factory(args)
    pprint(config.__dict__)
    if config.do_pre_processing:
        run_preprocessing(config)
    if config.do_create_dataset:
        create_dataset(config)


if __name__ == "__main__":
    main(parse_args())

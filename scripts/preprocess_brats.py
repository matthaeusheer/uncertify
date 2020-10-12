import os
import argparse
import glob
from tqdm import tqdm
from pathlib import Path
import subprocess
from pprint import pprint

import numpy as np
import h5py
import nibabel as nib
import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.data.preprocessing.histogram_matching.histogram_matching import MatchHistogramsTwoImages

VALID_MODALITIES = ['t1', 't2', 'flair', 't1ce', 'seg']
BACKGROUND_VALUE = -3.5
DEFAULT_BRATS_GLOB_PATH = f'/scratch/maheer/datasets/raw/BraTS2017/training/*/*'
HIST_REF_T1_PATH = f'/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t1_unbiased.nii.gz'
HIST_REF_T1_MASK_PATH = f'/scratch_net/samuylov/maheer/datasets/reference/Brats17_TCIA_607_1_t1_mask_unbiased.nii.gz'
HIST_REF_T2_PATH = f'to_be_done'
HIST_REF_T2_MASK_PATH = f'to_be_done'
HDF5_OUT_FOLDER = Path('/scratch/maheer/datasets/processed/')
DEFAULT_DATASET_NAME = 'BraTS17'
N4_EXECUTABLE = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4'


def run_preprocessing(brats_glob_path: str = DEFAULT_BRATS_GLOB_PATH, limit_n_samples: int = None,
                      process_t1w: bool = False, process_t2w: bool = False, process_segmentation: bool = True,
                      do_bias_correction: bool = False, force_bias_correction: bool = False,
                      do_hist_match: bool = False, do_normalization: bool = False,
                      normalization_method: str = 'rescale',
                      shuffle: bool = True, do_store_output: bool = True,
                      print_debug: bool = True) -> None:
    """Process BraTS dataset, performing histogram matching, normalization.

    Note:   This method creates additional output files in the original BraTS data folder. That is, for
            example, next to Brats17_TCIA_117_1_t1.nii.gz it will create Brats17_TCIA_117_1_t1_processed.nii.gz
            Those files are later read by the create_dataset function to create the HDF5 dataset.
    """
    sample_dir_paths = glob.glob(brats_glob_path)
    sample_dir_paths = [item for item in sample_dir_paths if 'Brats17_2013' not in os.path.split(item)[-1]]
    if shuffle:
        np.random.shuffle(sample_dir_paths)
    if limit_n_samples is not None:
        sample_dir_paths = sample_dir_paths[:limit_n_samples]

    for sample_dir_path in tqdm(sample_dir_paths, desc='Processing patients'):
        if print_debug:
            print(f'\n ------ Processing {sample_dir_path} ------')
        if process_t1w:
            process_modality(modality='t1',
                             sample_dir_path=sample_dir_path,
                             ref_img_path=HIST_REF_T1_PATH,
                             ref_mask_path=HIST_REF_T1_MASK_PATH,
                             do_hist_match=do_hist_match,
                             do_normalization=do_normalization,
                             normalization_method=normalization_method,
                             bias_correction=do_bias_correction,
                             force_bias_correction=force_bias_correction,
                             print_debug=print_debug,
                             do_store_output=do_store_output)
        if process_t2w:
            process_modality(modality='t2',
                             sample_dir_path=sample_dir_path,
                             ref_img_path=HIST_REF_T2_PATH,
                             ref_mask_path=HIST_REF_T2_MASK_PATH,
                             do_hist_match=do_hist_match,
                             do_normalization=do_normalization,
                             normalization_method=normalization_method,
                             bias_correction=do_bias_correction,
                             force_bias_correction=force_bias_correction,
                             print_debug=print_debug,
                             do_store_output=do_store_output)
        if process_segmentation:
            process_modality(modality='seg',
                             sample_dir_path=sample_dir_path,
                             do_hist_match=False,
                             do_normalization=False,
                             bias_correction=False,
                             force_bias_correction=force_bias_correction,
                             print_debug=print_debug,
                             do_store_output=do_store_output)


def process_modality(modality: str, sample_dir_path: str, ref_img_path: str = None, ref_mask_path: str = None,
                     do_hist_match: bool = False, do_normalization: bool = False, normalization_method: str = 'rescale',
                     do_store_output: bool = True, bias_correction: bool = True, force_bias_correction: bool = False,
                     print_debug: bool = True) -> None:
    """Process the images for a modality and return the processed images."""
    assert modality in VALID_MODALITIES, f'Invalid modality given {modality}!'
    if modality == 'seg':
        assert bias_correction is False, f'Do not attempt to run bias correction on a segmentation map, fool!'
    file_name_orig_glob = f'*{modality}.nii.gz'

    # Bias correction
    file_name_unbiased_glob = f"*{modality}_unbiased.nii.gz"
    if bias_correction and modality != 'seg':
        run_bias_correction(sample_dir_path, file_name_unbiased_glob, file_name_orig_glob,
                            force_bias_correction, print_debug)

    # Set file name we use to do further processing with
    file_name_glob = file_name_unbiased_glob if bias_correction else file_name_orig_glob
    nii_file_path = glob.glob(sample_dir_path + f'/' + file_name_glob)[0]  # Take first, only one should be found
    if print_debug:
        print(f'Processing {modality} slices: {nii_file_path}')

    # Load actual NII data file
    original_img: np.array = nib.load(nii_file_path).get_fdata()
    original_mask = create_masks(original_img)
    transformed_mask = create_masks(transform_image(original_img))

    # Histogram matching
    if do_hist_match:
        original_img = run_histogram_matching(original_img, original_mask, ref_img_path, ref_mask_path, print_debug)

    # Image transformation (transpose, rotate, crop, etc.)
    transformed_img = transform_image(original_img)

    # Normalization
    if do_normalization:
        transformed_img = normalize_image(transformed_img, transformed_mask, normalization_method,
                                          BACKGROUND_VALUE, print_debug)
    # Crush segmentation labels [1, 4] all to 1
    if modality == 'seg':
        transformed_img = np.where(transformed_img > 0, np.ones(transformed_img.shape), np.zeros(transformed_img.shape))

    # Storing output
    if do_store_output:
        store_pre_processing_output(transformed_img, transformed_mask, sample_dir_path, file_name_orig_glob,
                                    modality, print_debug)


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


def run_bias_correction(sample_dir_path: str, file_name_unbiased_glob: str,
                        file_name_orig_glob: str, force_bias_correction: bool, print_debug: bool = False) -> None:
    unbiased_glob_result = glob.glob(sample_dir_path + f'/' + file_name_unbiased_glob)
    if len(unbiased_glob_result) == 0:
        original_file_path = glob.glob(sample_dir_path + f'/' + file_name_orig_glob)[0]
        if print_debug:
            print(f'Running bias correction for {original_file_path}')
        run_bias_correction_single_file(Path(original_file_path), print_debug)
    else:
        if force_bias_correction:
            original_file_path = glob.glob(sample_dir_path + f'/' + file_name_orig_glob)[0]
            if print_debug:
                print(f'Running forced bias correction for {original_file_path}')
            run_bias_correction_single_file(Path(original_file_path), print_debug)
        else:
            if print_debug:
                print(f'Using existing unbiased file.')


def run_bias_correction_single_file(file_path: Path, print_debug: bool = False) -> None:
    """Run bias correction on a single file.

    Creates a new file in the same location with '<name>_unbiased.nii.gz'.
    Creates a new file in the same location for mask '<name>_mask_unbiased.nii.gz'"""
    in_file_name = file_path.name
    name_split = in_file_name.split('.', maxsplit=1)
    out_file_name = name_split[0] + '_unbiased.' + name_split[1]
    command = ' '.join([N4_EXECUTABLE, in_file_name, out_file_name])
    if print_debug:
        print(f'Running bias correction: {command}')
    subprocess.run(command, shell=True, cwd=file_path.parent)
    # Create a mask of the unbiased original slices
    img: np.array = nib.load(file_path).get_fdata()
    mask = create_masks(img)
    out_mask_file_name = name_split[0] + '_mask_unbiased.' + name_split[1]
    nif_mask_img = nib.Nifti1Image(mask, np.eye(4))
    nib.save(nif_mask_img, file_path.parent / out_mask_file_name)


def run_histogram_matching(img: np.ndarray, orig_mask: np.ndarray, ref_img_path: str, ref_mask_path: str,
                           print_debug: bool = False) -> np.ndarray:
    assert all([path is not None for path in [ref_img_path, ref_mask_path]]), f'Need to provide references for ' \
                                                                              f'histogram matching (mask & img)!'
    if print_debug:
        print(f'Performing histogram matching with {ref_img_path}')
    ref_img = nib.load(ref_img_path).get_fdata()
    ref_mask = nib.load(ref_mask_path).get_fdata()
    matched_img = MatchHistogramsTwoImages(ref_img, img, L=200, nbins=246, begval=0.05, finval=0.98,
                                           train_mask=ref_mask,
                                           test_mask=orig_mask)
    return matched_img


def store_pre_processing_output(img: np.ndarray, mask: np.ndarray, sample_dir_path: str, file_name_orig_glob: str,
                                modality: str, print_debug: bool = False) -> None:
    """Store the processed image slices and mask (both are transformed!) to the same folder as input."""
    nif_img = nib.Nifti1Image(img, np.eye(4))
    original_file_path = glob.glob(sample_dir_path + f'/' + file_name_orig_glob)[0]
    save_path = original_file_path.replace(f'{modality}.nii.gz', f'{modality}_processed.nii.gz')
    nib.save(nif_img, save_path)
    if print_debug:
        print(f'Saved {modality} sample to {save_path}.')
    if modality != 'seg':  # would output the segmentation mask...
        nif_mask = nib.Nifti1Image(mask, np.eye(4))
        save_path = original_file_path.replace(f'{modality}.nii.gz', 'mask.nii.gz')
        nib.save(nif_mask, save_path)
    if print_debug:
        print(f'Saved processed .nii.gz sample (and mask) to {save_path}')


def create_masks(slices: np.ndarray) -> np.ndarray:
    """Get the masks for (already manipulated) sample slices."""
    mask = (slices != 0).astype('int')
    return mask


def create_dataset(processed_glob_path: Path, hdf5_out_dir: Path, dataset_name: str, limit_n_samples: int = None,
                   do_t1: bool = False, do_t2: bool = False, do_seg: bool = False, print_debug: bool = True) -> None:
    """After pre-processing a whole dataset (writing processed images to disk), this function creates an HDF5
    dataset using h5py."""
    hdf5_out_dir.mkdir(parents=True, exist_ok=True)
    h5py_file = h5py.File(str(hdf5_out_dir / f'{dataset_name}.hdf5'), 'w')
    new_col = sum([do_t1, do_t2, do_seg]) + 1  # +1 for Mask

    sample_dir_paths = glob.glob(str(processed_glob_path))
    sample_dir_paths = [item for item in sample_dir_paths if 'Brats17_2013' not in os.path.split(item)[-1]]

    n_processed = 0
    for sample_dir_path in tqdm(sample_dir_paths, desc='Creating HDF dataset',
                                total=limit_n_samples if limit_n_samples is not None else len(sample_dir_paths)):
        if print_debug:
            print(f'\nProcessing {sample_dir_path}')
        # mask - do always
        mask_name = os.path.join(sample_dir_path, f"{os.path.split(sample_dir_path)[-1]}_mask.nii.gz")
        if print_debug:
            print(f'Loading mask: {mask_name}')
        mask_img = nib.load(mask_name).get_fdata()
        # TODO: Remove hardcoded 200*200 = 40000 size but make it dependent on width and height.
        mask_img = mask_img.reshape(-1, 200 * 200)
        if new_col:
            h5py_file.create_dataset('Mask', data=mask_img, maxshape=(None, 200 * 200))
            new_col = new_col - 1
        else:
            h5py_file["Mask"].resize((h5py_file["Mask"].shape[0] + len(mask_img)), axis=0)
            h5py_file["Mask"][-len(mask_img):] = mask_img

        if do_t1:
            t1_name = glob.glob(sample_dir_path + "/*t1_processed.nii.gz")[0]
            t1_img = nib.load(t1_name).get_fdata()
            t1_img = t1_img.reshape(-1, 200 * 200)
            if new_col:
                h5py_file.create_dataset('Scan', data=t1_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Scan"].resize((h5py_file["Scan"].shape[0] + len(t1_img)), axis=0)
                h5py_file["Scan"][-len(t1_img):] = t1_img

        if do_t2:
            t2_name = glob.glob(sample_dir_path + "/*t2_processed.nii.gz")[0]
            t2_img = nib.load(t2_name).get_fdata()
            t2_img = t2_img.reshape(-1, 200 * 200)
            if new_col:
                h5py_file.create_dataset('Scan_T2w', data=t2_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Scan_T2w"].resize((h5py_file["Scan_T2w"].shape[0] + len(t2_img)), axis=0)
                h5py_file["Scan_T2w"][-len(t2_img):] = t2_img

        if do_seg:
            seg_name = glob.glob(sample_dir_path + "/*seg_processed.nii.gz")[0]
            seg_img = nib.load(seg_name).get_fdata()

            seg_img = seg_img.reshape(-1, 200 * 200).astype(float)
            if new_col:
                h5py_file.create_dataset('Seg', data=seg_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Seg"].resize((h5py_file["Seg"].shape[0] + len(seg_img)), axis=0)
                h5py_file["Seg"][-len(seg_img):] = seg_img
        n_processed += 1
        if n_processed == limit_n_samples:
            break
    h5py_file_path = {h5py_file.filename}
    h5py_file.close()
    print(f'Done creating h5py dataset. Output: {h5py_file_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process a BraTS dataset and / or create a '
                                                 'HDF5 dataset out of it.')
    parser.add_argument('--name',
                        help='Name of the dataset created. E.g. "brats_val".')

    parser.add_argument('-p',
                        '--pre-process',
                        action='store_true',
                        help='If set, performs pre-processing of the whole dataset. '
                             'Output for the respective modalities will be stored at the same folder like the '
                             'original sample with "processed" prepended before the .nii.gz ending.')

    parser.add_argument('-g',
                        '--no-histogram-matching',
                        action='store_true',
                        help='Perform histogram matching vs a reference sample.')

    parser.add_argument('-n',
                        '--no-normalization',
                        action='store_true',
                        help='Perform zero mean / unit variance normalization.')

    parser.add_argument('--normalization-method',
                        type=str,
                        default='rescale',
                        choices=['rescale', 'standardize'],
                        help='How to normalize.')

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

    parser.add_argument('-s',
                        '--no-output',
                        action='store_true',
                        help='Skip storing output into "<name>_processed.nii.gz" files '
                             'at the same location during pre-processing step.')

    parser.add_argument('-l',
                        '--shuffle-pre-processing',
                        action='store_true',
                        help='Whether to shuffle the samples during pre-processing.')

    parser.add_argument('-t',
                        '--limit-n-samples',
                        type=int,
                        default=None,
                        help='Handy for debugging. Limits the processed samples to this number.')

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

    parser.add_argument('-d',
                        '--print-debug',
                        action='store_true',
                        help='If set, enables debug print information on stdout.')

    args = parser.parse_args()
    # Evaluation and Processing of args
    if not any([args.pre_process, args.create_dataset]):
        raise ValueError(f'Choose either -p (pre-process), -c (create-dataset) or both. You have not chosen any.')
    # Change the modalities args member such that it is a dict indicating whether to process it or not
    args.modalities = {modality: modality in args.modalities for modality in VALID_MODALITIES}
    return args


def main(args: argparse.Namespace) -> None:
    """We have to main """
    print(20 * '=')
    print(f'Processing BraTS dataset at {DEFAULT_BRATS_GLOB_PATH}')
    pprint(args.__dict__)
    print(20 * '=' + '\n')
    if args.pre_process:
        run_preprocessing(process_t1w=args.modalities['t1'],
                          process_t2w=args.modalities['t2'],
                          process_segmentation=args.modalities['seg'],
                          do_bias_correction=not args.no_bias_correction,
                          force_bias_correction=args.force_bias_correction,
                          do_hist_match=not args.no_histogram_matching,
                          do_normalization=not args.no_normalization,
                          normalization_method=args.normalization_method,
                          do_store_output=not args.no_output,
                          shuffle=args.shuffle_pre_processing,
                          limit_n_samples=args.limit_n_samples,
                          print_debug=args.print_debug)
    if args.create_dataset:
        create_dataset(processed_glob_path=Path(DEFAULT_BRATS_GLOB_PATH),
                       hdf5_out_dir=HDF5_OUT_FOLDER,
                       dataset_name=args.name,
                       limit_n_samples=args.limit_n_samples,
                       print_debug=args.print_debug,
                       do_t1=args.modalities['t1'],
                       do_t2=args.modalities['t2'],
                       do_seg=args.modalities['seg'])


if __name__ == "__main__":
    main(parse_args())

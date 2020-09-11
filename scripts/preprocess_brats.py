import argparse
import glob
from tqdm import tqdm
from pathlib import Path

import numpy as np
import h5py
import nibabel as nib
import add_uncertify_to_path  # makes sure we can use the uncertify library
from uncertify.data.preprocessing.histogram_matching.histogram_matching import MatchHistogramsTwoImages


VALID_MODALITIES = ['t1', 't2', 'flair', 't1ce', 'seg']
BACKGROUND_VALUE = -3.5
DEFAULT_BRATS_GLOB_PATH = f'/scratch/maheer/datasets/raw/BraTS2017/training/*/*'
HIST_REF_T1_PATH = f'/scratch/maheer/code/uncertify/data/brats_preprocessing/Brats17_TCIA_105_1_t1_unbiased.nii.gz'
HIST_REF_T1_MASK_PATH = f'/scratch/maheer/code/uncertify/data/brats_preprocessing/Brats17_TCIA_105_1_t1_unbiased.nii.gz'
HIST_REF_T2_PATH = f'to_be_done'
HIST_REF_T2_MASK_PATH = f'to_be_done'
HDF5_OUT_FOLDER = Path('/scratch/maheer/datasets/processed/')
DEFAULT_DATASET_NAME = 'BraTS17'


def run_preprocessing(brats_glob_path: str = DEFAULT_BRATS_GLOB_PATH, limit_n_samples: int = None,
                      process_t1w: bool = False, process_t2w: bool = False, process_segmentation: bool = True,
                      do_hist_match: bool = False, do_normalization: bool = False,
                      shuffle: bool = True, store_output: bool = True) -> None:
    """Process BraTS dataset, performing histogram matching, normalization.

    Note:   This method creates additional output files in the original BraTS data folder. That is, for
            example, next to Brats17_TCIA_117_1_t1.nii.gz it will create Brats17_TCIA_117_1_t1_processed.nii.gz
            Those files are later read by the create_dataset function to create the HDF5 dataset.
    """
    print(20 * '=')
    print(f'Processing BraTS dataset at {DEFAULT_BRATS_GLOB_PATH}')
    print(f'Modalities:')
    print(f'\tT1: {process_t1w}')
    print(f'\tT2: {process_t2w}')
    print(f'\tSeg: {process_segmentation}')
    print(20 * '=' + '\n')

    sample_dir_paths = glob.glob(brats_glob_path)
    if shuffle:
        np.random.shuffle(sample_dir_paths)
    if limit_n_samples is not None:
        sample_dir_paths = sample_dir_paths[:limit_n_samples]

    for sample_dir_path in tqdm(sample_dir_paths, desc='Processing patients'):
        print(f' ------ Processing {sample_dir_path} ------')
        if process_t1w:
            process_modality(modality='t1',
                             sample_dir_path=sample_dir_path,
                             ref_img_path=HIST_REF_T1_PATH,
                             ref_mask_path=HIST_REF_T1_MASK_PATH,
                             do_hist_match=do_hist_match,
                             do_normalization=do_normalization,
                             store_output=store_output)
        if process_t2w:
            process_modality(modality='t2',
                             sample_dir_path=sample_dir_path,
                             ref_img_path=HIST_REF_T2_PATH,
                             ref_mask_path=HIST_REF_T2_MASK_PATH,
                             do_hist_match=do_hist_match,
                             do_normalization=do_normalization,
                             store_output=store_output)
        if process_segmentation:
            process_modality(modality='seg',
                             sample_dir_path=sample_dir_path,
                             do_hist_match=False,
                             do_normalization=False,
                             store_output=store_output)


def process_modality(modality: str, sample_dir_path: str, ref_img_path: str = None, ref_mask_path: str = None,
                     do_hist_match: bool = False, do_normalization: bool = False,
                     store_output: bool = True) -> None:
    """Process the images for a modality and return the processed images."""
    assert modality in VALID_MODALITIES, f'Invalid modality given {modality}!'
    nii_file_path = glob.glob(sample_dir_path + f"/*{modality}.nii.gz")[0]
    print(f'Processing {modality} slices: {nii_file_path}')
    img: np.array = nib.load(nii_file_path).get_data()
    img = manipulate_sample(img)
    mask = create_masks(img)
    if do_hist_match:
        assert all([path is not None for path in [ref_img_path, ref_mask_path]]), f'Need to provide references for ' \
                                                                                  f'histogram matching (mask & img)!'
        print(f'Performing histogram matching with {ref_img_path}')
        ref_img = nib.load(ref_img_path).get_data()
        ref_mask = nib.load(ref_mask_path).get_data()

        img = MatchHistogramsTwoImages(ref_img, img, L=200, nbins=246, begval=0.05,
                                       finval=0.98, train_mask=ref_mask,
                                       test_mask=mask)
    if do_normalization:
        print(f'Performing normalization (zero mean, unit variance), set background to {BACKGROUND_VALUE}.')
        img = (img - img[mask != 0].mean()) / img[mask != 0].std()
        img[mask == 0] = BACKGROUND_VALUE

    if store_output:
        nif_img = nib.Nifti1Image(img, np.eye(4))
        save_path = nii_file_path.replace(f'{modality}.nii.gz', f'{modality}_processed.nii.gz')
        nib.save(nif_img, save_path)
        if modality != 'seg':  # would output the segmentation mask...
            nif_mask = nib.Nifti1Image(mask, np.eye(4))
            save_path = nii_file_path.replace(f'{modality}.nii.gz', 'mask.nii.gz')
            nib.save(nif_mask, save_path)
        print(f'Saved processed .nii.gz sample (and mask) to {save_path}')
    return


def manipulate_sample(slices: np.ndarray) -> np.ndarray:
    """Do all manipulations to the raw numpy array like transposing, rotation etc."""
    slices = np.transpose(slices, axes=[2, 0, 1])  # makes dimensions to be [slice, width, height]?
    slices = np.rot90(slices, k=1, axes=(2, 1))  # rotates once in the (2, 1) plane, i.e. width-height-plane
    slices = slices[:, 27:227, 20:220]  # <-- why???
    return slices


def create_masks(slices: np.ndarray) -> np.ndarray:
    """Get the masks for (already manipulated) sample slices."""
    mask = (slices != 0).astype('int')
    return mask


def create_dataset(processed_glob_path: Path, hdf5_out_dir: Path, dataset_name: str, limit_n_samples: int = None,
                   do_t1: bool = False, do_t2: bool = False, do_seg: bool = False) -> None:
    """After pre-processing a whole dataset (writing processed images to disk), this function creates an HDF5
    dataset using h5py."""
    hdf5_out_dir.mkdir(parents=True, exist_ok=True)
    h5py_file = h5py.File(str(hdf5_out_dir / f'{dataset_name}.hdf5'), 'w')
    new_col = sum([do_t1, do_t2, do_seg]) + 1  # +1 for Mask

    sample_dir_paths = sorted(glob.glob(str(processed_glob_path)))
    if limit_n_samples is not None:
        sample_dir_paths = sample_dir_paths[:limit_n_samples]

    for sample_dir_path in tqdm(sample_dir_paths, desc='Processing samples'):
        if do_t1:
            t1_name = glob.glob(sample_dir_path + "/*t1_processed.nii.gz")[0]
            t1_img = nib.load(t1_name).get_data()
            t1_img = t1_img.reshape(-1, 200 * 200)
            if new_col:
                h5py_file.create_dataset('Scan', data=t1_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Scan"].resize((h5py_file["Scan"].shape[0] + len(t1_img)), axis=0)
                h5py_file["Scan"][-len(t1_img):] = t1_img

        if do_t2:
            t2_name = glob.glob(sample_dir_path + "/*t2_processed.nii.gz")[0]
            t2_img = nib.load(t2_name).get_data()
            t2_img = t2_img.reshape(-1, 200 * 200)
            if new_col:
                h5py_file.create_dataset('Scan_T2w', data=t2_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Scan_T2w"].resize((h5py_file["Scan_T2w"].shape[0] + len(t2_img)), axis=0)
                h5py_file["Scan_T2w"][-len(t2_img):] = t2_img

        if do_seg:
            seg_name = glob.glob(sample_dir_path + "/*seg_processed.nii.gz")[0]
            seg_img = nib.load(seg_name).get_data()

            seg_img = seg_img.reshape(-1, 200 * 200)
            if new_col:
                h5py_file.create_dataset('Seg', data=seg_img, maxshape=(None, 200 * 200))
                new_col = new_col - 1
            else:
                h5py_file["Seg"].resize((h5py_file["Seg"].shape[0] + len(seg_img)), axis=0)
                h5py_file["Seg"][-len(seg_img):] = seg_img
        # mask - do always
        mask_name = glob.glob(sample_dir_path + "/*mask.nii.gz")[0]
        mask_img = nib.load(mask_name).get_data()
        mask_img = mask_img.reshape(-1, 200 * 200)
        if new_col:
            h5py_file.create_dataset('Mask', data=mask_img, maxshape=(None, 200 * 200))
            new_col = new_col - 1
        else:
            h5py_file["Mask"].resize((h5py_file["Mask"].shape[0] + len(mask_img)), axis=0)
            h5py_file["Mask"][-len(mask_img):] = mask_img

    h5py_file.close()
    print('Done creating h5py dataset.')


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process a BraTS dataset and / or create a '
                                                 'HDF5 dataset out of it.')
    parser.add_argument('-p',
                        '--pre-process',
                        action='store_true',
                        default=False,
                        help='If set, performs pre-processing of the whole dataset. '
                             'Output for the respective modalities will be stored at the same folder like the '
                             'original sample with "processed" prepended before the .nii.gz ending.')
    parser.add_argument('-c',
                        '--create-dataset',
                        action='store_true',
                        default=False,
                        help='If set, creates an HDF5 dataset out of the processed dataset (see -p above).')
    args = parser.parse_args()
    if not any([args.pre_process, args.create_dataset]):
        raise ValueError(f'Choose either -p (pre-process), -c (create-dataset) or both. You have not chosen any.')
    return args


def main(args: argparse.Namespace) -> None:
    # NOTE: At the moment the parameters are hard-coded, s.t. only T1 samples are used in the HDF5 dataset.
    if args.pre_process:
        run_preprocessing(process_t1w=True,
                          process_t2w=False,
                          process_segmentation=True,
                          do_hist_match=False,
                          do_normalization=True,
                          store_output=True)
    if args.create_dataset:
        create_dataset(processed_glob_path=Path(DEFAULT_BRATS_GLOB_PATH),
                       hdf5_out_dir=HDF5_OUT_FOLDER,
                       dataset_name=DEFAULT_DATASET_NAME,
                       limit_n_samples=10,
                       do_t1=True,
                       do_t2=False,
                       do_seg=True)


if __name__ == "__main__":
    main(parse_args())

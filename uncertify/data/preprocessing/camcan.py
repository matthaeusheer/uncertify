from pathlib import Path


from typing import List, Optional


MODALITY_MAP = {'t1': 'T1w', 't2': 'T2w'}


def get_camcan_nii_sample_file_paths(root_dir_path: Path, modality: str,
                                     keyword: Optional[str] = 'unbiased') -> List[Path]:
    """Gets all nii file paths for a given modality.
    Assumes that the folder layout is as follows
        CamCAN/
            T1w/
                <sample_id>_T1w_unbiased.nii.gz
                    ...
                <sample_id>T1w_brain_mask.nii.gz
                    ...
            T2w/
                <sample_id>_T2w_unbiased.nii.gz
                    ...
                <sample_id>T2w_brain_mask.nii.gz
                    ...
    Arguments:
        root_dir_path: path of the directory in which the 'T1w' and 'T2w' folders reside.
        modality: imaging modality, 't1' or 't2'
        keyword: only if this keyword is in a file name the file is added to the list returned by this function
    """
    sample_dir_path = root_dir_path / MODALITY_MAP[modality]
    paths = [path for path in sample_dir_path.iterdir()]
    if keyword is not None:
        paths = [path for path in paths if keyword in path.name]
    return paths


def get_camcan_nii_mask_file_paths(root_dir_path: Path, modality: str, keyword: str = 'mask') -> List[Path]:
    """Works similar to get_nii_sample_files_paths but for masks."""
    sample_dir_path = root_dir_path / MODALITY_MAP[modality]
    paths = [path for path in sample_dir_path.iterdir() if keyword in path.name]
    return paths

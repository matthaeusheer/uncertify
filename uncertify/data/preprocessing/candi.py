from pathlib import Path

from typing import List


def get_candi_sample_dir_paths(root_dir_path: Path) -> List[Path]:
    """Get a list of sample dir paths, one sample corresponds to one full scan.
    Arguments:
        root_dir_path: path to the location where there are sample directories at this location
    """
    return list(filter(Path.is_dir, root_dir_path.iterdir()))


def get_candi_nii_file_paths(dir_paths: List[Path], file_selector: str) -> List[Path]:
    """Returns all the .nii.gz file paths for a given file_selector type.
    Arguments:
        dir_paths: a list of sample dir paths, each directory holds a full scan
        file_selector: a string representing which file type to chose, e.g. 'ana' for IBSR_02_ana.nii.gz
    """
    nii_file_paths = []
    for dir_path in dir_paths:
        sample_name = dir_path.name
        file_name = f'{sample_name}{file_selector}.nii.gz'
        file_path = dir_path / file_name
        nii_file_paths.append(file_path)
    return nii_file_paths

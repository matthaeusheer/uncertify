from pathlib import Path
from typing import List

import h5py


def print_dataset_information(dataset_paths: List[Path]) -> h5py.File:
    for path in dataset_paths:
        print(f'{path} does{" not " if not path.exists() else " "}exist!')

    def print_datasets_info(h5py_file: h5py.File) -> None:
        for dataset_name, dataset in h5py_file.items():
            print(dataset)

    for path in dataset_paths:
        name = path.name
        h5py_file = h5py.File(path, 'r')
        print(f'\n --- {name} ---')
        print_datasets_info(h5py_file)
    print('Metadata:')
    for key, val in h5py_file.attrs.items():
        print(f'\t{key:30}: {val}')
    return h5py_file

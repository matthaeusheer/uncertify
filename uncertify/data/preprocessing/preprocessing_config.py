import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

VALID_BRATS_MODALITIES = ['t1', 't2', 'flair', 't1ce', 'seg']
BACKGROUND_VALUE = -3.5
N4_EXECUTABLE_PATH = Path('/usr/bmicnas01/data-biwi-01/bmicdatasets/Sharing/N4')


@dataclass
class PreprocessConfig:
    """A single config used throughout a processing run."""
    dataset_name: str
    dataset_root_path: Path
    limit_to_n_samples: int
    exclude_empty_slices: bool
    do_histogram_matching: bool
    ref_paths: Dict[str, Dict[str, Path]]
    do_normalization: bool
    normalization_method: str
    background_value: float
    hdf5_out_folder_path: Path
    n4_executable_path: Path
    print_debug: bool


@dataclass
class BratsConfig(PreprocessConfig):
    do_pre_processing: bool
    do_create_dataset: bool
    modalities: Dict[str, bool]
    do_bias_correction: bool
    force_bias_correction: bool
    do_normalization: bool
    normalization_method: str
    shuffle_pre_processing: bool
    store_pre_processing_output: bool

    @property
    def image_modality(self) -> str:
        modalities_dict = self.modalities
        # TODO: This is rather dangerous.
        true_modalities = [key for key in modalities_dict.keys() if key != 'seg' and modalities_dict[key] is True]
        assert len(true_modalities) == 1, f'Error, multiple image modalities not allowed at once!'
        return true_modalities[0]


@dataclass
class CamCanConfig(PreprocessConfig):
    image_modality: str


def preprocess_config_factory(args: argparse.Namespace, ref_paths: dict,
                              dataset_type: str) -> Union[BratsConfig, CamCanConfig]:
    """Factory method to create a pre-processing config based on the parsed command line arguments."""
    if dataset_type == 'brats':
        config = BratsConfig(
            dataset_name=args.dataset_name,
            dataset_root_path=args.dataset_root_path,
            do_pre_processing=args.pre_process,
            do_create_dataset=args.create_dataset,
            modalities={modality: modality in args.modalities for modality in VALID_BRATS_MODALITIES},
            limit_to_n_samples=args.limit_n_samples,
            exclude_empty_slices=args.exclude_empty_slices,
            do_bias_correction=not args.no_bias_correction,
            force_bias_correction=args.force_bias_correction,
            do_histogram_matching=not args.no_histogram_matching,
            ref_paths=ref_paths,
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
    elif dataset_type == 'camcan':
        config = CamCanConfig(
            dataset_name=args.dataset_name,
            dataset_root_path=args.dataset_root_path,
            image_modality=args.modality,
            limit_to_n_samples=args.limit_n_samples,
            exclude_empty_slices=args.exclude_empty_slices,
            do_histogram_matching=not args.no_histogram_matching,
            ref_paths=ref_paths,
            do_normalization=not args.no_normalization,
            normalization_method=args.normalization_method,
            background_value=BACKGROUND_VALUE,
            hdf5_out_folder_path=args.hdf5_out_dir_path,
            n4_executable_path=N4_EXECUTABLE_PATH,
            print_debug=args.print_debug
        )
        return config


def validate_preprocess_config(config: PreprocessConfig) -> PreprocessConfig:
    """Validates that the config can produce a valid pre-processing run and return the valid config."""
    # TODO: Code this up.
    return config

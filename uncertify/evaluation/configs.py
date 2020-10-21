import json

from dataclasses import dataclass
from pathlib import Path

from uncertify.data.preprocessing.preprocessing_config import PreprocessConfig


@dataclass
class PixelThresholdSearchConfig:
    accepted_fpr: 0.05
    # Min, max, and number of values for fpr vs threshold calculations
    min_val = 0.0
    max_val = 1.0
    num_values = 5
    num_batches = 10
    # Subsequent Golden Section Search parameters
    gss_lower_val = 0.0
    gss_upper_val = 1.0
    gss_tolerance = 0.003


@dataclass
class PerformanceEvaluationConfig:
    use_n_batches: int = None


@dataclass
class EvaluationConfig:
    """A data class which holds configurations and hyper-parameters necessary for an evaluation pipeline run."""
    train_dataset_config: PreprocessConfig
    test_dataset_config: PreprocessConfig
    thresh_search_config: PixelThresholdSearchConfig
    performance_config: PerformanceEvaluationConfig

@dataclass
class EvaluationResult:
    """A data class holding every single result which comes out of a whole evaluation pipeline run."""
    # The input evaluation config defining all hyper parameters for complete evaluation run
    evaluation_config: EvaluationConfig

    # Some path and file name stuff
    out_dir_path: Path
    run_dir_name_prefix: str = 'evaluation_'
    plot_dir_name: str = 'plots'
    img_dir_name: str = 'images'

    # Residual pixel threshold calculation
    best_threshold: float = None

    # Segmentation performance
    dice_score_global: float = None
    per_patient_dice_score_mean: float = None
    per_patient_dice_score_std: float = None

    # Pixel-wise anomaly detection performance
    au_prc: float = None
    au_roc: float = None

    def make_dirs(self) -> None:
        """Creates all output directories needed to story results."""
        (self.out_dir_path / self.run_dir_name).mkdir()
        for dir_path in {self.plot_dir_path, self.img_dir_path}:
            dir_path.mkdir()

    @property
    def run_dir_name(self) -> str:
        """Output of different runs will be stored in self.run_dir_name_prefix_<run_id> folders. This function returns
        the ful run directory name for a run based on what run folders have already been created."""
        run_dir_names = [path.name for path in self.out_dir_path.iterdir()
                         if path.is_dir() and self.run_dir_name_prefix in path.name]
        if len(run_dir_names) == 0:
            current_run_number = 0
        else:
            current_run_number = max(int(name.split('_')[-1]) for name in run_dir_names) + 1
        return f'{self.run_dir_name_prefix}_{current_run_number}'

    @property
    def plot_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.plot_dir_name

    @property
    def img_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.img_dir_name

    def to_json(self) -> None:
        """Store the whole dataclass to disk (in out_dir_path) as a .json file."""
        with open(self.out_dir_path / self.run_dir_name / 'evaluation_results.json') as outfile:
            json.dump(self.__dict__, outfile, indent=4, ensure_ascii=False)

    def to_pickle(self) -> None:
        # TODO: Implement
        pass

    def from_pickle(self) -> None:
        # TODO: Implement
        pass
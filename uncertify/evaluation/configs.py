import json
import logging

from dataclasses import dataclass, field
from pathlib import Path

from uncertify.data.preprocessing.preprocessing_config import PreprocessConfig
from uncertify.evaluation.inference import SliceWiseCriteria

from typing import Union, List

LOG = logging.getLogger(__name__)


@dataclass
class PixelThresholdSearchConfig:
    # Accepted false positive rate when searching for threshold
    accepted_fpr: float = 0.1
    # Min, max, and number of values for fpr vs threshold calculations
    min_val = 0.0
    max_val = 4.0
    num_values = 7
    # Subsequent Golden Section Search parameters
    gss_lower_val = 0.0
    gss_upper_val = 10.0
    gss_tolerance = 0.001


@dataclass
class PerformanceEvaluationConfig:
    # Min, max, and number of values for segmentation performance vs threshold calculations
    min_val = 0.0
    max_val = 3.0
    num_values = 5
    do_iou = False
    do_multiple_thresholds = False


@dataclass
class EvaluationConfig:
    """A data class which holds configurations and hyper-parameters necessary for an evaluation pipeline run."""
    thresh_search_config: PixelThresholdSearchConfig() = PixelThresholdSearchConfig()
    performance_config: PerformanceEvaluationConfig = PerformanceEvaluationConfig()
    use_n_batches: int = None  # Handy for faster evaluation or debugging
    train_dataset_config: PreprocessConfig = None
    test_dataset_config: PreprocessConfig = None
    do_plots: bool = False


@dataclass
class AnomalyDetectionResult:
    au_prc: float = 0.0
    au_roc: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self) -> str:
        return f'AU_ROC {self.au_roc:.2f}, AU_PRC: {self.au_prc:.2f}'


@dataclass
class PixelAnomalyDetectionResult(AnomalyDetectionResult):
    # Residual pixel threshold calculation
    best_threshold: float = None

    # Segmentation performance at best threshold
    per_patient_dice_score_mean: float = None
    per_patient_dice_score_std: float = None

    def to_dict(self) -> dict:
        out_dict = {key: value for key, value in self.__dict__.items()}
        out_dict.update(super().to_dict())
        return out_dict

    def __str__(self) -> str:
        return f'{super().__str__()}, Dice(mean/std): ' \
               f'({self.per_patient_dice_score_mean:.2f}, {self.per_patient_dice_score_std:.2f}) '


@dataclass
class OODDetectionResult(AnomalyDetectionResult):
    mode: str = None  # can be 'combo', 'lesional', 'healthy'

    def to_dict(self) -> dict:
        out_dict = {key: value for key, value in self.__dict__.items()}
        out_dict.update(super().to_dict())
        return out_dict


@dataclass
class OODDetectionResults:
    results: List[OODDetectionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'results': [res.to_dict() for res in self.results]}


@dataclass
class SliceAnomalyDetectionResult(AnomalyDetectionResult):
    criteria: SliceWiseCriteria = None

    def to_dict(self) -> dict:
        out_dict = {key: value for key, value in self.__dict__.items()}
        out_dict.update(super().to_dict())
        return out_dict

    def __str__(self) -> str:
        return f'Criteria {self.criteria}, {super().__str__()}'


@dataclass
class SliceAnomalyDetectionResults:
    results: List[SliceAnomalyDetectionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'results': [res.to_dict() for res in self.results]}

    def __str__(self) -> str:
        return f'\n'.join([str(result) for result in self.results])


@dataclass
class EvaluationResult:
    """A data class holding every single result which comes out of a whole evaluation pipeline run."""
    # Some path and file name stuff
    out_dir_path: Union[Path, str]
    pixel_anomaly_result: PixelAnomalyDetectionResult
    slice_anomaly_results: SliceAnomalyDetectionResults
    ood_detection_results: OODDetectionResults
    run_dir: str = None
    plot_dir_name: str = 'plots'
    img_dir_name: str = 'images'

    @property
    def current_run_number(self) -> int:
        """Output of different runs will be stored in self.run_dir_name_prefix_<run_id> folders. This function returns
        the ful run directory name for a run based on what run folders have already been created."""
        if self.out_dir_path.exists():
            run_dir_names = [path.name for path in self.out_dir_path.iterdir()
                             if path.is_dir() and self.run_dir_name_prefix in path.name]
        else:
            run_dir_names = []
        if len(run_dir_names) == 0:
            current_run_number = 0
        else:
            current_run_number = max(int(name.split('_')[-1]) for name in run_dir_names)
        return current_run_number

    def make_dirs(self) -> None:
        """Creates all output directories needed to story results."""
        (self.out_dir_path / self.run_dir_name).mkdir(parents=True)
        LOG.info(f'Created evaluation run directory: {self.out_dir_path / self.run_dir_name}')
        for dir_path in {self.plot_dir_path, self.img_dir_path}:
            dir_path.mkdir()

    @property
    def run_dir_name_prefix(self) -> str:
        return 'evaluation_'

    @property
    def run_dir_name(self) -> str:
        if self.run_dir is None:
            self.run_dir = f'{self.run_dir_name_prefix}{self.current_run_number + 1}'
        return self.run_dir

    @property
    def plot_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.plot_dir_name

    @property
    def img_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.img_dir_name

    def to_dict(self) -> dict:
        dicts_list = [self.pixel_anomaly_result.to_dict(),
                      self.slice_anomaly_results.to_dict(),
                      self.ood_detection_results.to_dict()]
        names = ['pixel_anomaly_result', 'slice_anomaly_results', 'ood_detection_result']
        all_dicts = {}
        for name, name_dict in zip(names, dicts_list):
            all_dicts[name] = name_dict
        return all_dicts

    def to_json(self) -> None:
        """Store the whole dataclass to disk (in out_dir_path) as a .json file."""
        with open(Path(self.out_dir_path) / self.run_dir_name / 'evaluation_results.json', 'w') as outfile:
            self.out_dir_path = str(self.out_dir_path)  # for json to work
            json.dump(self.to_dict(), outfile, indent=4, ensure_ascii=False)

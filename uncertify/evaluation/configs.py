import json
import logging

from dataclasses import dataclass, field, asdict
from pathlib import Path

from uncertify.data.preprocessing.preprocessing_config import PreprocessConfig
from uncertify.evaluation.inference import SliceWiseCriteria

from typing import Union, List

LOG = logging.getLogger(__name__)


@dataclass
class PixelThresholdSearchConfig:
    # Accepted false positive rate when searching for threshold
    accepted_fpr: float = 0.05
    # Min, max, and number of values for fpr vs threshold calculations for plotting
    min_val: float = 0.1
    max_val: float = 1.0
    num_values: int = 10
    # Subsequent Golden Section Search parameters
    gss_lower_val: float = 0.0
    gss_upper_val: float = 2.00
    gss_tolerance: float = 0.001


@dataclass
class PerformanceEvaluationConfig:
    # Min, max, and number of values for segmentation performance vs threshold calculations
    min_val: float = 0.3
    max_val: float = 1.5
    num_values: int = 10
    do_iou: bool = False
    do_multiple_thresholds: bool = False


@dataclass
class OodEvaluationConfig:
    metrics: tuple = ('dose', )  # 'waic'
    dose_statistics: tuple = ('rec_err', 'kl_div', 'elbo', 'entropy') #, 'kl_div', 'elbo', 'entropy')  # ('rec_err', 'kl_div', 'elbo')


@dataclass
class EvaluationConfig:
    """A data class which holds configurations and hyper-parameters necessary for an evaluation pipeline run."""
    thresh_search_config: PixelThresholdSearchConfig = PixelThresholdSearchConfig()
    seg_performance_config: PerformanceEvaluationConfig = PerformanceEvaluationConfig()
    ood_config: OodEvaluationConfig = OodEvaluationConfig()
    use_n_batches: int = None  # Handy for faster evaluation or debugging
    train_dataset_config: PreprocessConfig = None
    test_dataset_config: PreprocessConfig = None
    do_plots: bool = False
    use_masked_loss: bool = True


@dataclass
class AnomalyDetectionResult:
    au_prc: float = None
    au_roc: float = None

    def __str__(self) -> str:
        au_roc_formatted = f'{self.au_roc:.3f}' if self.au_roc else None
        au_prc_formatted = f'{self.au_prc:.3f}' if self.au_prc else None
        return f'AU_ROC {au_roc_formatted}, AU_PRC: {au_prc_formatted}'


@dataclass
class PixelAnomalyDetectionResult(AnomalyDetectionResult):
    # Residual pixel threshold calculation
    best_threshold: float = None

    # Segmentation performance at best threshold
    per_patient_dice_score_mean: float = None
    per_patient_dice_score_std: float = None

    def __str__(self) -> str:
        return f'{super().__str__()}, Dice(mean/std): ' \
               f'({self.per_patient_dice_score_mean:.3f}, {self.per_patient_dice_score_std:.3f}) '


@dataclass
class OODDetectionResult(AnomalyDetectionResult):
    mode: str = None  # can be 'all', 'lesional', 'healthy'
    metric: str = None  # e.g. WAIC or DoSE


@dataclass
class OODDetectionResults:
    results: List[OODDetectionResult] = field(default_factory=list)


@dataclass
class SliceAnomalyDetectionResult(AnomalyDetectionResult):
    criteria: SliceWiseCriteria = None

    def __str__(self) -> str:
        return f'Criteria {self.criteria}, {super().__str__()}'


@dataclass
class SliceAnomalyDetectionResults:
    results: List[SliceAnomalyDetectionResult] = field(default_factory=list)

    def __str__(self) -> str:
        return f'\n'.join([str(result) for result in self.results])


@dataclass
class EvaluationResult:
    """A data class holding every single result which comes out of a whole evaluation pipeline run."""
    # Some path and file name stuff
    out_dir_path: Union[Path, str]
    evaluation_config: EvaluationConfig
    pixel_anomaly_result: PixelAnomalyDetectionResult
    slice_anomaly_results: SliceAnomalyDetectionResults
    ood_detection_results: OODDetectionResults
    run_dir: str = None
    plot_dir_name: str = 'plots'
    img_dir_name: str = 'images'
    train_set_name: str = 'Train Set'
    test_set_name: str = 'Test Set'
    comments: List[str] = field(default_factory=list)

    @property
    def _current_run_number(self) -> int:
        """Output of different runs will be stored in self.run_dir_name_prefix_<run_id> folders. This function returns
        the run number based on what run folders have already been created."""
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

    @property
    def run_number(self) -> int:
        """The actual run number of THIS run."""
        return int(self.run_dir_name.split('_')[-1])

    def make_dirs(self) -> None:
        """Creates all output directories needed to story results."""
        (self.out_dir_path / self.run_dir_name).mkdir(parents=True)
        for dir_path in {self.plot_dir_path, self.img_dir_path}:
            dir_path.mkdir()

    @property
    def run_dir_name_prefix(self) -> str:
        return 'evaluation_'

    @property
    def run_dir_name(self) -> str:
        if self.run_dir is None:
            self.run_dir = f'{self.run_dir_name_prefix}{self._current_run_number + 1}'
        return self.run_dir

    @property
    def plot_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.plot_dir_name

    @property
    def img_dir_path(self) -> Path:
        return self.out_dir_path / self.run_dir_name / self.img_dir_name

    def store_json(self) -> None:
        """Store the whole dataclass to disk (in out_dir_path) as a .json file."""
        out_file_path = Path(self.out_dir_path) / self.run_dir_name / f'evaluation_results_{self._current_run_number}.json'
        with open(out_file_path, 'w') as outfile:
            self.out_dir_path = str(self.out_dir_path)  # for json to work
            json.dump(asdict(self), outfile, indent=4, ensure_ascii=False)

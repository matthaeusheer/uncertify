from pathlib import Path
from typing import List

from torch import nn
from torch.utils.data import DataLoader

from uncertify.common import DATA_DIR_PATH
from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult, PixelAnomalyDetectionResult, \
    SliceAnomalyDetectionResults, OODDetectionResults
from uncertify.evaluation.evaluation_pipeline import run_ood_detection_performance, print_results_from_evaluation_dirs
from uncertify.evaluation.ood_metrics import LOG, sample_wise_waic_scores


def run_ood_evaluations(train_dataloader: DataLoader, dataloader_dict: dict, ensemble_models: List[nn.Module],
                        residual_threshold: float = None, max_n_batches: int = None,
                        eval_results_dir: Path = DATA_DIR_PATH / 'evaluation',
                        print_results_when_done: bool = True, ) -> None:
    """Run OOD evaluation pipeline for multiple dataloaders amd store output in evaluation output directory.

    Arguments
        train_dataloader: the in-distribution dataloader from training the model
        dataloader_dict: (name, dataloader) dictionary
        ensemble_models: a list of trained ensemble models
        residual_threshold: as usual
        eval_results_dir: for each dataloader, i.e. each run an output folder with results will be created here
        print_results_when_done: self-explanatory
    """
    run_numbers = []
    for name, dataloader in dataloader_dict.items():
        LOG.info(f'OOD evaluation for {name}...')
        eval_cfg = EvaluationConfig()
        eval_cfg.do_plots = True
        eval_cfg.use_n_batches = max_n_batches
        results = EvaluationResult(eval_results_dir, eval_cfg, PixelAnomalyDetectionResult(),
                                   SliceAnomalyDetectionResults(), OODDetectionResults())
        results.pixel_anomaly_result.best_threshold = residual_threshold
        results.make_dirs()
        run_numbers.append(results.run_number)
        run_ood_detection_performance(ensemble_models, train_dataloader, dataloader, eval_cfg, results)
        results.test_set_name = name
        results.to_json()
    if print_results_when_done:
        print_results_from_evaluation_dirs(eval_results_dir, run_numbers, print_results_only=True)


def run_ood_to_ood_dict(dataloader_dict: dict, ensemble_models: list, num_batches: int = None) -> dict:
    """Run OOD detection and organise output such that healthy vs. lesional analysis can be performed.

    # TODO: Define an interface for functions that return slice-wise ood scores and take the function as
            an input parameter s.t. this works for different OOD metrics.
    Returns
        ood_dict: one sub-dict for every dataloader where the name is the key and for the sub-dicts the
                  keys are ['all', 'healthy', 'lesional', 'healthy_scans', 'lesional_scans' where the values for the
                  scans keys are slice-wise pytorch tensors and the rest are actual OOD scores in a list either
                  all of them, only the healthy ones or only lesional ones
    """
    ood_dict = {}
    for name, data_loader in dataloader_dict.items():
        LOG.info(f'WAIC score calculation for {name} ({num_batches * data_loader.batch_size} slices)...')
        slice_wise_waic_scores, slice_wise_is_lesional, scans = sample_wise_waic_scores(models=ensemble_models,
                                                                                        data_loader=data_loader,
                                                                                        max_n_batches=num_batches,
                                                                                        return_slices=True)
        # Organise as healthy / unhealthy
        healthy_scores = []
        lesional_scores = []
        healthy_scans = []
        lesional_scans = []
        for idx in range(len(slice_wise_waic_scores)):
            is_lesional_slice = slice_wise_is_lesional[idx]
            if is_lesional_slice:
                lesional_scores.append(slice_wise_waic_scores[idx])
                if scans is not None:
                    lesional_scans.append(scans[idx])
            else:
                healthy_scores.append(slice_wise_waic_scores[idx])
                if scans is not None:
                    healthy_scans.append(scans[idx])

        dataset_ood_dict = {'all': slice_wise_waic_scores, 'healthy': healthy_scores, 'lesional': lesional_scores,
                            'healthy_scans': healthy_scans, 'lesional_scans': lesional_scans}
        ood_dict[name] = dataset_ood_dict
    return ood_dict

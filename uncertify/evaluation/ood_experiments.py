from collections import defaultdict
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from uncertify.common import DATA_DIR_PATH
from uncertify.evaluation.configs import EvaluationConfig, EvaluationResult, PixelAnomalyDetectionResult, \
    SliceAnomalyDetectionResults, OODDetectionResults
from uncertify.evaluation.evaluation_pipeline import run_ood_detection_performance, print_results_from_evaluation_dirs
from uncertify.evaluation.waic import LOG, sample_wise_waic_scores
from uncertify.evaluation.dose import full_pipeline_slice_wise_dose_scores
from uncertify.evaluation.statistics import aggregate_slice_wise_statistics, fit_statistics

from typing import Tuple, List


def run_ood_evaluations(train_dataloader: DataLoader, dataloader_dict: dict, ensemble_models: List[nn.Module],
                        residual_threshold: float = None, max_n_batches: int = None,
                        eval_results_dir: Path = DATA_DIR_PATH / 'evaluation',
                        print_results_when_done: bool = True) -> None:
    """Run OOD evaluation pipeline for multiple dataloaders amd store output in evaluation output directory. There
    will be one evaluation output directory per OOD dataloader.

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


def run_ood_to_ood_dict(test_dataloader_dict: dict, ensemble_models: list, train_dataloader: DataLoader = None,
                        num_batches: int = None, ood_metrics: Tuple[str] = ('waic', 'dose'),
                        dose_statistics: Tuple[str] = None) -> dict:
    """Run OOD detection and organise output such that healthy vs. lesional analysis can be performed.

    Returns
        ood_dict: top-level keys are ood metrics, then we have a dict with
                  one sub-dict for every dataloader where the name is the key and for the sub-dicts the
                  keys are ['all', 'healthy', 'lesional', 'healthy_scans', 'lesional_scans' where the values for the
                  scans keys are slice-wise pytorch tensors and the rest are actual OOD scores in a list either
                  all of them, only the healthy ones or only lesional ones
    """
    # Fit training distribution beforehand, if dose is requested
    kde_func_dict = None
    if 'dose' in ood_metrics:
        assert dose_statistics is not None, f'Need to provide DoSE statistics.'
        model = ensemble_models[0]
        statistics_dict = aggregate_slice_wise_statistics(model, train_dataloader, dose_statistics,
                                                          max_n_batches=num_batches)
        kde_func_dict = fit_statistics(statistics_dict)

    metrics_ood_dict = defaultdict(dict)
    for metric in ood_metrics:
        for name, data_loader in test_dataloader_dict.items():
            LOG.info(f'{metric} score calculation for {name} ({num_batches * data_loader.batch_size} slices)...')
            if metric == 'waic':
                slice_wise_ood_scores, slice_wise_is_lesional, scans = sample_wise_waic_scores(models=ensemble_models,
                                                                                               data_loader=data_loader,
                                                                                               max_n_batches=num_batches,
                                                                                               return_slices=True)
            elif metric == 'dose':
                slice_wise_ood_scores, dose_kde_dict = full_pipeline_slice_wise_dose_scores(train_dataloader,
                                                                                            data_loader,
                                                                                            ensemble_models[0],
                                                                                            dose_statistics,
                                                                                            max_n_batches=num_batches,
                                                                                            kde_func_dict=kde_func_dict)
                slice_wise_is_lesional = dose_kde_dict['is_lesional']
                scans = dose_kde_dict['scans']
            else:
                raise ValueError(f'Requested OOD metric {metric} not supported.')

            # Organise as healthy / unhealthy
            healthy_scores = []
            lesional_scores = []
            healthy_scans = []
            lesional_scans = []
            for idx in range(len(slice_wise_ood_scores)):
                is_lesional_slice = slice_wise_is_lesional[idx]
                if is_lesional_slice:
                    lesional_scores.append(slice_wise_ood_scores[idx])
                    if scans is not None:
                        lesional_scans.append(scans[idx])
                else:
                    healthy_scores.append(slice_wise_ood_scores[idx])
                    if scans is not None:
                        healthy_scans.append(scans[idx])

            dataset_ood_dict = {'all': slice_wise_ood_scores, 'healthy': healthy_scores, 'lesional': lesional_scores,
                                'healthy_scans': healthy_scans, 'lesional_scans': lesional_scans}
            metrics_ood_dict[metric][name] = dataset_ood_dict
    return metrics_ood_dict

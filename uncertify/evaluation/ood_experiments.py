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

from typing import Tuple, List, Union


def run_ood_evaluations(train_dataloader: DataLoader, dataloader_dict: dict,
                        models_or_model: Union[List[nn.Module], nn.Module],
                        residual_threshold: float = None, max_n_batches: int = None,
                        eval_results_dir: Path = DATA_DIR_PATH / 'evaluation',
                        print_results_when_done: bool = True) -> None:
    """Run OOD evaluation pipeline for multiple dataloaders amd store output in evaluation output directory. There
    will be one evaluation output directory per OOD dataloader.

    Arguments
        train_dataloader: the in-distribution dataloader from training the model
        dataloader_dict: (name, dataloader) dictionary
        models_or_model: a list of trained ensemble models or a single model, note that e.g. WAIC needs ensembles
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
        LOG.info(f'Evaluation run number: {results.run_number}')
        run_numbers.append(results.run_number)
        run_ood_detection_performance(models_or_model, train_dataloader, dataloader, eval_cfg, results)
        results.test_set_name = name
        results.store_json()
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

    metrics_ood_dict = defaultdict(dict)
    for metric in ood_metrics:
        for name, test_data_loader in test_dataloader_dict.items():
            LOG.info(f'{metric} ODD score calculation for {name}')
            if metric == 'waic':
                waic_results = sample_wise_waic_scores(models=ensemble_models,
                                                       data_loader=test_data_loader,
                                                       max_n_batches=num_batches,
                                                       return_slices=True)
                slice_wise_ood_scores = waic_results.slice_wise_waic_scores
                slice_wise_is_lesional = waic_results.slice_wise_is_lesional
                scans = waic_results.slice_wise_scans
                segmentations = waic_results.slice_wise_segmentations
                masks = waic_results.slice_wise_masks
                reconstructions = waic_results.slice_wise_reconstructions
            elif metric == 'dose':
                # Fit training distribution beforehand, if dose is requested
                assert dose_statistics is not None, f'Need to provide DoSE statistics.'
                model = ensemble_models[0]  # no need for ensemble models in DoSE, simply use first
                LOG.info(f'Fitting data statistics on {train_dataloader.dataset.name}')
                statistics_dict = aggregate_slice_wise_statistics(model, train_dataloader, dose_statistics,
                                                                  max_n_batches=num_batches)
                kde_func_dict = fit_statistics(statistics_dict)

                slice_wise_ood_scores, dose_kde_dict, test_stat_dict = full_pipeline_slice_wise_dose_scores(
                    train_dataloader,
                    test_data_loader,
                    model,
                    dose_statistics,
                    max_n_batches=num_batches,
                    kde_func_dict=kde_func_dict)

                slice_wise_is_lesional = test_stat_dict['is_lesional']
                scans = test_stat_dict['scans']
                segmentations = test_stat_dict['segmentations']
                masks = test_stat_dict['masks']
                reconstructions = test_stat_dict['reconstructions']
            else:
                raise ValueError(f'Requested OOD metric {metric} not supported.')

            # Organise as healthy / unhealthy scores and images
            final_scores = defaultdict(list)
            final_scans = defaultdict(list)
            final_segmentations = defaultdict(list)
            final_reconstructions = defaultdict(list)

            # For DoSE track organized individual dose scores as well
            dose_kde_healthy = defaultdict(list)
            dose_kde_lesional = defaultdict(list)
            dose_stat_healthy = defaultdict(list)
            dose_stat_lesional = defaultdict(list)

            for idx in range(len(slice_wise_ood_scores)):
                lesional_or_healthy_str = 'lesional' if slice_wise_is_lesional[idx] else 'healthy'

                final_scores[lesional_or_healthy_str].append(slice_wise_ood_scores[idx])
                final_scans[lesional_or_healthy_str].append(scans[idx])
                final_segmentations[lesional_or_healthy_str].append(segmentations[idx])
                if reconstructions is not None:
                    final_reconstructions[lesional_or_healthy_str].append(reconstructions[idx])

                if metric == 'dose':  # Track dose KDE and statistics values
                    for statistic in dose_statistics:
                        if slice_wise_is_lesional[idx]:
                            dose_stat_lesional[statistic].append(test_stat_dict[statistic][idx])
                            dose_kde_lesional[statistic].append(dose_kde_dict[statistic][idx])
                        else:
                            dose_stat_healthy[statistic].append(test_stat_dict[statistic][idx])
                            dose_kde_healthy[statistic].append(dose_kde_dict[statistic][idx])

            dataset_ood_dict = {'all': slice_wise_ood_scores,
                                'masks': masks,
                                'healthy': final_scores['healthy'],
                                'lesional': final_scores['lesional'],
                                'healthy_scans': final_scans['healthy'],
                                'lesional_scans': final_scans['lesional'],
                                'healthy_segmentations': final_segmentations['healthy'],
                                'lesional_segmentations': final_segmentations['lesional'],
                                'healthy_reconstructions': final_reconstructions['healthy'],
                                'lesional_reconstructions': final_reconstructions['lesional'],
                                'dose_kde_healthy': dose_kde_healthy,
                                'dose_kde_lesional': dose_kde_lesional,
                                'dose_stat_healthy': dose_stat_healthy,
                                'dose_stat_lesional': dose_stat_lesional}
            metrics_ood_dict[metric][name] = dataset_ood_dict
    return metrics_ood_dict

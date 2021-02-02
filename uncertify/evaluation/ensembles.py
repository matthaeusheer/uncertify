import torch
import torchvision
import matplotlib.pyplot as plt

from uncertify.models.vae import VariationalAutoEncoder
from uncertify.data.dataloaders import DataLoader
from uncertify.evaluation.inference import yield_inference_batches, BatchInferenceResult
from uncertify.evaluation.utils import mask_background_to_value
from uncertify.visualization.grid import imshow_grid, imshow

from typing import List, Generator, Tuple

BACKGROUND_VAL = -3.5
V_MAX = 4
V_MIN = BACKGROUND_VAL


def infer_ensembles(ensemble_models: List[VariationalAutoEncoder], dataloader: DataLoader,
                    use_n_batches: int, residual_threshold: float, **kwargs) -> List:
    """Runs inference using every single model of the ensembles and yields the results from every single one.

    Yields:
        result_generators: one result_generator per model over which one can iterate to get inference batch results
    """
    for model in ensemble_models:
        result_generator = yield_inference_batches(dataloader, model, use_n_batches, residual_threshold, **kwargs)
        yield result_generator


def combine_ensemble_results(model_results_generator: List[Generator]) -> Generator:
    """Create a zipped batch results tuple for multiple result generators.
    Arguments:
        model_results_generator: as returned by infer_ensembles, a list of generators which produce batch results
    Yields:
        results: tuples of BatchInferenceResult's where each entry corresponds to result from one ensemble model
    """
    yield from zip(*model_results_generator)


def visualize_ensemble_predictions(ensemble_results: Generator[Tuple[BatchInferenceResult], None, None],
                                   **kwargs) -> None:
    for result_tuple in ensemble_results:
        mask = result_tuple[0].mask
        scan = mask_background_to_value(result_tuple[0].scan, mask, BACKGROUND_VAL)

        vert_stacked_reconstructions = torch.cat(
            [mask_background_to_value(result.reconstruction, mask, BACKGROUND_VAL) for result in result_tuple],
            dim=2)

        stacked_all = torch.cat((scan, vert_stacked_reconstructions), dim=2)

        reconstruction_grid = torchvision.utils.make_grid(stacked_all, padding=0, normalize=False)

        imshow_grid(reconstruction_grid, vmin=V_MIN, vmax=V_MAX, **kwargs)

        visualize_mean_std_prediction(result_tuple, **kwargs)
        visualize_ensemble_residuals(result_tuple, **kwargs)


def visualize_mean_std_prediction(result_tuple: Tuple[BatchInferenceResult], **kwargs) -> None:
    mask = result_tuple[0].mask
    stacked_reconstructions = torch.stack(
        [mask_background_to_value(result.reconstruction, mask, value=BACKGROUND_VAL) for result in result_tuple])
    mean_reconstruction = torch.mean(stacked_reconstructions, dim=0)
    std_reconstruction = mask_background_to_value(torch.std(stacked_reconstructions, dim=0), mask, value=0)

    # mean
    imshow_grid(torchvision.utils.make_grid(mean_reconstruction, padding=0), **kwargs, vmin=V_MIN, vmax=V_MAX)
    # standard deviation
    kwargs.update({'cmap': 'hot'})
    imshow_grid(torchvision.utils.make_grid(std_reconstruction, padding=0), **kwargs)


def visualize_ensemble_residuals(result_tuple: Tuple[BatchInferenceResult], **kwargs) -> None:
    """Given a tuple of BatchInferenceResult's, plot the mean and std of the residual map."""
    mask = result_tuple[0].mask
    stacked_residuals = torch.stack(
        [mask_background_to_value(result.residual, mask, value=0) for result in result_tuple]
    )
    mean_residuals = torch.mean(stacked_residuals, dim=0)
    std_residuals = mask_background_to_value(torch.std(stacked_residuals, dim=0), mask, value=0)

    stacked_all = torch.cat((mean_residuals, std_residuals), dim=2)

    grid = torchvision.utils.make_grid(stacked_all, padding=0, normalize=False)
    kwargs['cmap'] = 'afmhot'
    imshow_grid(grid, **kwargs)

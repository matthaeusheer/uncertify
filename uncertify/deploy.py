import math
import logging
import itertools

import scipy.stats
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from uncertify.utils.custom_types import Tensor
from uncertify.evaluation.utils import threshold_batch_to_one_zero
from uncertify.utils.tensor_ops import normalize_to_0_1
from uncertify.utils.tensor_ops import print_scipy_stats_description
from uncertify.data.utils import get_n_masked_pixels

from typing import Generator, Dict, Tuple, Callable

LOG = logging.getLogger(__name__)


def l1_distance(t1: Tensor, t2: Tensor) -> Tensor:
    return torch.abs(t2 - t1)


def yield_reconstructed_batches(data_loader: DataLoader,
                                trained_model: torch.nn.Module,
                                max_batches: int = None,
                                residual_threshold: float = None,
                                residual_fn: Callable = l1_distance,
                                get_batch_fn: Callable = lambda batch: batch['scan'],
                                print_statistics: bool = False,
                                progress_bar_suffix: str = '') -> Generator[Dict[str, Tensor], None, None]:
    """For some dataloader and a trained model, run the 'scan' tensors of the dataloader through the model
    and yield a tuple dicts of scan, reconstruction and (if present in dataloader) segmentation batches."""
    data_generator = itertools.islice(data_loader, max_batches) if max_batches is not None else data_loader
    n_batches = max_batches if max_batches is not None else len(data_loader)
    for batch in tqdm(data_generator, desc=f'Inferring batches {progress_bar_suffix}', total=n_batches):
        scan_batch = get_batch_fn(batch)
        # Run actual inference for batch
        inference_result = trained_model(scan_batch)
        reconstruction_batch, mu, log_var, total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err, latent_code = inference_result

        # Add image tensors to output
        residual_batch = residual_fn(reconstruction_batch, scan_batch)
        thresholded_batch = normalize_to_0_1(residual_batch)
        if residual_threshold is not None:
            thresholded_batch = threshold_batch_to_one_zero(thresholded_batch, residual_threshold)
        output = {'scan': scan_batch, 'rec': reconstruction_batch, 'res': residual_batch, 'thresh': thresholded_batch,
                  'latent_code': latent_code}
        try:
            if 'seg' in batch.keys():
                # add segmentation if available
                output['seg'] = torch.where(batch['seg'] > 0,
                                            torch.ones_like(batch['seg']),
                                            torch.zeros_like(batch['seg']))
            if 'mask' in batch.keys():
                output['mask'] = batch['mask']
        except AttributeError as error:
            # LOG.warning(f'Batch has no keys. Probably MNIST like. {error}')
            pass

        # Add loss terms to output
        for key, value in zip(['total_loss', 'mean_kld_div', 'mean_rec_err', 'kl_div', 'rec_err'],
                              [total_loss, mean_kld_div, mean_rec_err, kl_div, rec_err]):
            output[key] = value

        # Print statistics
        if print_statistics:
            with torch.no_grad():
                for name, sub_batch in output.items():
                    description = scipy.stats.describe(sub_batch.detach().numpy().flatten())
                    print_scipy_stats_description(name=name, description_result=description)
        yield output


def yield_y_true_y_pred(data_loader: DataLoader,
                        trained_model: torch.nn.Module,
                        max_n_batches: int = None,
                        residual_fn: Callable = l1_distance,
                        print_statistics: bool = False):
    """Yield flattened vectors over all (of max_n_batches, if not None) batches for y_true and y_pred.
    For Args: see yield_reconstruction_batches. Similar.
    Yields:
        tuple of y_true and y_pred, where
            y_true: a one-hot encoded vector where 1 stands for abnormal pixel and 0 stands for normal pixel
            y_pred: the residual vector (values ranging from 0 to 1)
    """
    # For further calculations (sklearn) need ground truth (n) - array and prediction probabilities as (n, 2) - array
    # where column 0 is normal probability and column 1 is abnormal probability (1-normal_probability)
    # Per batch one list which will hold an array for the per-batch ground truth and prediction masked pixels
    y_true_batch_list = []
    y_pred_batch_list = []

    for batch_idx, batch in enumerate(yield_reconstructed_batches(data_loader=data_loader,
                                                                  trained_model=trained_model,
                                                                  max_batches=max_n_batches,
                                                                  residual_threshold=None,
                                                                  residual_fn=residual_fn,
                                                                  print_statistics=print_statistics)):
        with torch.no_grad():
            # Get ground truth and prediction for whole brain
            batch_y_true = batch['seg'][batch['mask']].flatten().numpy()
            abnormal_pred = normalize_to_0_1(batch['res'][batch['mask']]).flatten().numpy()
            normal_pred = 1 - abnormal_pred
            stacked = np.vstack((normal_pred, abnormal_pred))
            y_true_batch_list.append(batch_y_true)
            y_pred_batch_list.append(stacked)
    y_true_np = np.hstack(y_true_batch_list)
    y_pred_np = np.concatenate(y_pred_batch_list, axis=1)
    return y_true_np, y_pred_np.T


def sample_from_gauss_prior(n_samples: int, latent_space_dim: int) -> Tensor:
    """Returns a (n_samples, latent_space_dim)-shaped tensor with samples from the Gaussian prior latent space."""
    return torch.normal(mean=0, std=torch.ones((n_samples, latent_space_dim)))


def infer_latent_space_samples(model: torch.nn.Module, latent_samples: Tensor) -> Tensor:
    """Run inference only on the decoder part of the model using some samples from the latent space."""
    with torch.no_grad():
        return model._decoder(latent_samples)

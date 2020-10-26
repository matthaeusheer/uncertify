from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
import umap

from uncertify.deploy import infer_latent_space_samples
from uncertify.visualization.reconstruction import plot_vae_output
from uncertify.visualization.plotting import save_fig, setup_plt_figure
from uncertify.evaluation import latent_space_analysis
from uncertify.utils.custom_types import Tensor

from typing import Tuple, Iterable, List, Generator, Dict


def plot_umap_latent_embedding(output_generators: Iterable[Generator[Dict[str, Tensor], None, None]],
                               names: Iterable[str], **kwargs) -> plt.Figure:
    """For different data loaders, plot the latent space UMAP embedding.

    Arguments:
        output_generators: an iterable of generators (use yield_reconstructed_batches) to create histograms for
        names: names for the respective output generators

    """
    # Prepare dictionaries holding sample-wise codes for different data loaders
    latent_codes = defaultdict(list)
    kl_divs = defaultdict(list)
    rec_errors = defaultdict(list)

    # Perform inference and fill the data dicts for later plotting
    for generator, generator_name in zip(output_generators, names):
        for batch in generator:
            has_ground_truth = 'seg' in batch.keys()
            if has_ground_truth:
                for segmentation, mask, kl_div, rec_err, code in zip(batch['seg'], batch['mask'],
                                                                     batch['kl_div'], batch['rec_err'],
                                                                     batch['latent_code']):  # sample-wise zip
                    n_abnormal_pixels = float(torch.sum(segmentation > 0))
                    n_normal_pixels = float(torch.sum(mask))
                    if n_normal_pixels == 0:
                        continue
                    abnormal_fraction = n_abnormal_pixels / n_normal_pixels
                    is_abnormal = n_abnormal_pixels > 20  # abnormal_fraction > 0.01  # TODO: Remove hardcoded.
                    suffix = 'abnormal' if is_abnormal else 'normal'

                    kl_divs[f' '.join([generator_name, suffix])].append(float(kl_div))
                    rec_errors[f' '.join([generator_name, suffix])].append(float(rec_err))
                    latent_codes[f' '.join([generator_name, suffix])].append(code.detach().numpy())
            else:
                for kl_div, rec_err, code in zip(batch['kl_div'], batch['rec_err'],
                                                 batch['latent_code']):  # sample-wise zip
                    kl_divs[generator_name].append(float(kl_div))
                    rec_errors[generator_name].append(float(rec_err))
                    latent_codes[generator_name].append(code.detach().numpy())
    kld_arrays = []
    kld_labels = []
    rec_arrays = []
    rec_labels = []
    code_arrays = []
    code_labels = []

    for name, arr in kl_divs.items():
        kld_arrays.append(np.array(arr))
        kld_labels.append(name)
    for name, arr in rec_errors.items():
        rec_arrays.append(np.array(rec_errors[name]))
        rec_labels.append(name)
    for name, arr in latent_codes.items():
        code_arrays.append(np.array(arr))
        code_labels.append(name)

    sizes = np.concatenate(kld_arrays)

    # Perform UMAP embedding
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(np.vstack(code_arrays))

    fig, ax = setup_plt_figure(title='Latent Space UMAP Embeddings', **kwargs)

    unique_names = {name: idx for idx, name in enumerate(set(code_labels))}
    sample_lengths = {name: len(codes) for name, codes in zip(code_labels, code_arrays)}
    color_indices = []
    for name, length in sample_lengths.items():
        color_indices.extend(length * [unique_names[name]])

    pointer = 0
    for dataset_name, dataset_length in sample_lengths.items():
        x_values = embedding[pointer:pointer+dataset_length, 0]
        y_values = embedding[pointer:pointer+dataset_length, 1]

        ax.scatter(
            x_values,
            y_values,
            c=[sns.color_palette()[unique_names[dataset_name]]] * dataset_length,
            s=sizes[pointer:pointer+dataset_length] * 1000,
            label=dataset_name,
            alpha=0.7,
            edgecolors='none',
        )
        pointer += dataset_length

    ax.legend()
    ax.set_aspect('equal', 'datalim')
    return fig


def plot_latent_reconstructions_one_dim_changing(trained_model: torch.nn.Module, change_dim_idx: int, n_samples: int,
                                                 save_path: Path = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Sampling (0, 0, ..., 0) in latent space and then varying one single dimension and check the impact."""
    latent_samples = latent_space_analysis.get_latent_samples_one_dim(dim_idx=change_dim_idx,
                                                                      n_samples=n_samples)
    output = infer_latent_space_samples(trained_model, latent_samples)
    fig, ax = plot_vae_output(output, figsize=(20, 20), vmax=3, one_channel=True, axis='off', nrow=n_samples,
                              add_colorbar=False, **kwargs)
    if save_path is not None:
        save_fig(fig, save_path, **kwargs)
    return fig, ax


def plot_latent_reconstructions_multiple_dims(model: torch.nn.Module, latent_space_dims: int = 128,
                                              n_samples_per_dim: int = 16,
                                              save_path: Path = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Plot latent space sample reconstructions for multiple dimensions, i.e. cover all cases where all dimensions
    but one are fixed."""
    outputs = []
    sample_generator = latent_space_analysis.yield_samples_all_dims(latent_space_dims, n_samples_per_dim)
    for bp in tqdm(sample_generator, desc='Inferring latent space samples', total=latent_space_dims):
        outputs.append(infer_latent_space_samples(model, bp))
    stacked = torch.cat(outputs, dim=0)
    fig, ax = plot_vae_output(stacked, figsize=(20, 20), one_channel=True, axis='off', nrow=n_samples_per_dim,
                              transpose_grid=True, add_colorbar=False, vmax=3, **kwargs)
    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_latent_reconstructions_2d_grid(model: torch.nn.Module, dim1: int, dim2: int,
                                        grid_size: int = 16, save_path: Path = None,
                                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    outputs = []
    sample_generator = latent_space_analysis.latent_space_2d_grid(dim1=dim1, dim2=dim2, grid_size=grid_size)
    for bp in tqdm(sample_generator, desc='Inferring latent space samples', total=16):
        outputs.append(infer_latent_space_samples(model, bp))
    stacked = torch.cat(outputs, dim=0)
    fig, ax = plot_vae_output(stacked, figsize=(20, 20), one_channel=True, axis='off', nrow=grid_size,
                              transpose_grid=False,
                              add_colorbar=False, vmax=3, **kwargs)
    if save_path is not None:
        save_fig(fig, save_path)
    return fig, ax


def plot_random_latent_space_samples(model: torch.nn.Module, n_samples: int = 16,
                                     latent_space_dims: int = 128, save_path: Path = None,
                                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Sample zero mean unit variance samples from latent space and visualize."""
    latent_samples = torch.normal(mean=0, std=torch.ones((n_samples, latent_space_dims,)))
    output = infer_latent_space_samples(model, latent_samples)
    fig, ax = plot_vae_output(output, figsize=(20, 20), one_channel=True, axis='off',
                              add_colorbar=False, vmax=3, **kwargs)
    if save_path is not None:
        save_fig(fig, save_path, **kwargs)
    return fig, ax

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from uncertify.deploy import infer_latent_space_samples
from uncertify.visualization.reconstruction import plot_vae_output
from uncertify.visualization.plotting import save_fig
from uncertify.evaluation import latent_space_analysis

from typing import Tuple


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

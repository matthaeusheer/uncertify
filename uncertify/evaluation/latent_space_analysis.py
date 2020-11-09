import numpy as np
import torch

from uncertify.utils.custom_types import Tensor

from typing import Generator


def yield_samples_all_dims(latent_space_dim: int = 128, n_samples_per_dim: int = 64) -> Generator[Tensor, None, None]:
    for dim_idx in range(latent_space_dim):
        yield get_latent_samples_one_dim(dim_idx, n_samples_per_dim)


def get_latent_samples_one_dim(dim_idx: int, n_samples: int = 64, latent_space_dim: int = 128,
                               z_min: float = -3, z_max: float = 3) -> Tensor:
    latent_samples = torch.zeros((n_samples, latent_space_dim))
    latent_samples[:, dim_idx] = torch.linspace(z_min, z_max, n_samples)
    return latent_samples


def latent_space_2d_grid(dim1: int, dim2: int, latent_space_dim: int = 128, grid_size: int = 16,
                         z_min: float = -3, z_max: float = 3) -> Generator[Tensor, None, None]:
    grid_values = np.linspace(z_min, z_max, grid_size)
    for row_idx in range(grid_size):
        row = torch.zeros((grid_size, latent_space_dim))
        for col_idx in range(grid_size):
            row[col_idx][dim1] = grid_values[row_idx]  # row-wise change for dim1
        for col_idx in range(grid_size):
            row[col_idx][dim2] = grid_values[col_idx]  # col-wise change for dim2
        yield row


def sample_from_gauss_prior(n_samples: int, latent_space_dim: int) -> Tensor:
    """Returns a (n_samples, latent_space_dim)-shaped tensor with samples from the Gaussian prior latent space."""
    return torch.normal(mean=0, std=torch.ones((n_samples, latent_space_dim)))
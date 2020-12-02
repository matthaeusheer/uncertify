"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import dataclasses
from pathlib import Path
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
import torch.distributions as dist

from uncertify.models.gradient import Gradient
from uncertify.models.beta_annealing import monotonic_annealing, cyclical_annealing, sigmoid_annealing, decay_annealing
from uncertify.models.beta_annealing import BetaConfig, ConstantBetaConfig, MonotonicBetaConfig, \
    CyclicBetaConfig, SigmoidBetaConfig, DecayBetaConfig
from uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.utils.sampling import random_uniform_ring
from uncertify.evaluation.utils import residual_l1_max
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Dict

LATENT_SPACE_DIM = 128


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 beta_config: BetaConfig = ConstantBetaConfig(beta=1.0),
                 n_m_factor: float = 1.0, ood_dataloaders: List[Dict[str, DataLoader]] = None,
                 loss_type: str = 'l1') -> None:
        """Variational Auto Encoder which works with generic encoders and decoders.
        Arguments:
            encoder: the encoder module (can be pytorch or lightning module)
            decoder: the decoder module (can be pytorch or lightning module)
            beta_config: configuration on how to handle the beta parameter (constant, monotonic or cyclic)
            ood_dataloaders: a list of dataloaders which will be used to check reconstructions of during training
            loss_type: can be either "l1" or "l2" to use either Laplace or Gaussian distributions over reconstruction
        """
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._gradient_net = Gradient()
        self._train_step_counter = 0
        self._beta_config = beta_config
        # self.save_hyperparameters('decoder', 'encoder')
        self._n_m_factor = n_m_factor
        self._ood_dataloaders = ood_dataloaders
        self._loss_type = loss_type

    def forward(self, img_tensor: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor,
                                                                 Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            img_tensor: raw image tensor (un-flattened)
            mask: the brain mask
        """
        mu, log_var = self._encoder(img_tensor)
        latent_code = self._reparameterize(mu, log_var)
        reconstruction = self._decoder(latent_code)
        total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err = self.loss_function(reconstruction, img_tensor, mask,
                                                                                    mu, log_var)
        return reconstruction, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code

    @staticmethod
    def _reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Applying re-parameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(std^2) = 2*log(std) -> std = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']
        mask = batch['mask']

        # Run batch through model and get losses
        rec_batch, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(features,
                                                                                                           mask)

        # Log training losses to Tensorboard
        train_loss_terms = {'train_reconstruction_error': mean_rec_err,
                            'train_kl_div': mean_kl_div}
        self.logger.experiment.add_scalar('train_total_loss', total_loss,
                                          global_step=self._train_step_counter)
        self.logger.experiment.add_scalars('train_loss_term', train_loss_terms,
                                           global_step=self._train_step_counter)
        self.logger.experiment.add_scalar('beta', self._calculate_beta(self._train_step_counter),
                                          global_step=self._train_step_counter)
        self._train_step_counter += 1
        return {'loss': total_loss}  # w.r.t this will be optimized

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']
        mask = batch['mask']

        # Run validation batch through model and get losses
        rec_batch, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(features,
                                                                                                           mask)
        # Those values will be tracked and can be accessed in validation_epoch_end
        val_return_dict = {'val_total_loss': total_loss,
                           'val_kl_div': mean_kl_div,
                           'val_rec_err': mean_rec_err,
                           'reconstructed_batch': rec_batch,
                           'input_batch': features}
        return val_return_dict

    def validation_epoch_end(self, outputs: List[Dict]) -> dict:
        # Compute average validation losses and log
        avg_val_total_losses = torch.stack([x['val_total_loss'] for x in outputs])
        avg_val_kld_losses = torch.stack([x['val_kl_div'] for x in outputs])
        avg_val_recon_losses = torch.stack([x['val_rec_err'] for x in outputs])
        self.logger.experiment.add_scalar('avg_val_mean_total_loss', avg_val_total_losses.mean(),
                                          global_step=self._train_step_counter)
        loss_terms = {'avg_val_mean_kl_div': avg_val_kld_losses.mean(),
                      'avg_val_mean_rec_err': avg_val_recon_losses.mean()}
        self.logger.experiment.add_scalars('avg_val_mean_loss_terms', loss_terms, global_step=self._train_step_counter)

        # Sample from the same few batches over training and check reconstructions
        self._log_random_eval_batch_reconstruction(outputs)

        # Sample from latent space and log reconstructions, from unit Gaussian and from within ranges
        for min_max_range in [None, (1, 2), (3, 4), (6, 7), (20, 30), (50, 100), (100, 150)]:
            self._log_latent_samples_reconstructions(n_samples=16, min_max_radius=min_max_range)

        # Sample from OOD datasets if provided
        if self._ood_dataloaders is not None:
            for ood_set in self._ood_dataloaders:
                for ood_set_name, ood_dataloader in ood_set.items():
                    self._log_ood_reconstructions(ood_set_name, ood_dataloader)

        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param.data, global_step=self._train_step_counter)

        loss_terms.update({'avg_val_mean_total_loss': avg_val_total_losses.mean()})
        return loss_terms

    def loss_function(self, reconstruction: Tensor, observation: Tensor, mask: Tensor, mu: Tensor, log_var: Tensor,
                      beta: float = 1.0, train_step: int = None):
        if self._loss_type == 'l2':
            rec_dist = dist.Normal(reconstruction, 1.0)
        elif self._loss_type == 'l1':
            rec_dist = dist.Laplace(reconstruction, 1.0)
        else:
            raise ValueError(f'Loss type "{self._loss_type}" not supported.')

        """
        # p(x|z)
        log_p_x_z = rec_dist.log_prob(observation)
        log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))  # Per slice negative log likelihood
        slice_rec_err = log_p_x_z

        # p(z)
        z_prior = dist.Normal(0.0, 1.0)

        # q(z|x)
        z_post = dist.Normal(mu, torch.sqrt(torch.exp(log_var)))

        # KL(q(z|x), p(z))
        slice_kl_div = dist.kl_divergence(z_post, z_prior)
        slice_kl_div = torch.mean(slice_kl_div, dim=1)  # KL divergences for each slice
        slice_kl_div = self._calculate_beta(self._train_step_counter) * slice_kl_div

        # Take the mean over all batches to get batch-wise kl divergence and log likelihood compared to slice-wise
        batch_kl_div = torch.mean(slice_kl_div)
        batch_rec_err = torch.mean(log_p_x_z)

        variational_lower_bound = (-batch_kl_div + batch_rec_err) * self._n_m_factor
        total_loss = -variational_lower_bound
        """

        mask = torch.ones_like(reconstruction) if mask is None else mask

        # Reconstruction Error
        slice_rec_err = torch_functional.l1_loss(observation*mask, reconstruction*mask, reduction='none')
        slice_rec_err = torch.sum(slice_rec_err, dim=(1, 2, 3))
        slice_wise_non_empty = torch.sum(mask, dim=(1, 2, 3))
        slice_wise_non_empty[slice_wise_non_empty == 0] = 1  # for empty masks
        slice_rec_err /= slice_wise_non_empty
        batch_rec_err = torch.mean(slice_rec_err)

        # KL Divergence
        slice_kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        batch_kl_div = torch.mean(slice_kl_div)
        kld_loss = batch_kl_div * self._calculate_beta(self._train_step_counter)

        # Total Loss
        total_loss = batch_rec_err + kld_loss

        return total_loss, batch_kl_div, batch_rec_err, slice_kl_div, slice_rec_err

    def configure_optimizers(self) -> Optimizer:
        """Pytorch-lightning function."""
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def _calculate_beta(self, train_step: int) -> float:
        """Calculates the KL term for any strategy for a given training_step"""
        if isinstance(self._beta_config, ConstantBetaConfig):
            return self._beta_config.beta
        elif isinstance(self._beta_config, MonotonicBetaConfig):
            return monotonic_annealing(train_step, **dataclasses.asdict(self._beta_config))
        elif isinstance(self._beta_config, CyclicBetaConfig):
            return cyclical_annealing(train_step, **dataclasses.asdict(self._beta_config))
        elif isinstance(self._beta_config, SigmoidBetaConfig):
            return sigmoid_annealing(train_step, **dataclasses.asdict(self._beta_config))
        elif isinstance(self._beta_config, DecayBetaConfig):
            return decay_annealing(train_step, **dataclasses.asdict(self._beta_config))
        else:
            raise RuntimeError(f'Beta config not supported.')

    def _log_latent_samples_reconstructions(self, n_samples: int, min_max_radius: Tuple[float, float] = None) -> None:
        """Sample from latent space and check reconstructions."""
        with torch.no_grad():
            if min_max_radius is None:  # sample from standard normal gaussian
                latent_samples = torch.normal(mean=0, std=torch.ones((n_samples, LATENT_SPACE_DIM,))).cuda()
                log_title = 'ID_mu0_std1_latent_samples'
            else:  # sample from a certain "hyper-circle" around the normal mode
                inner_radius, outer_radius = min_max_radius
                latent_samples = torch.tensor(random_uniform_ring(np.zeros(128), outer_radius,
                                                                  inner_radius, n_samples)).float().cuda()
                log_title = f'ID_range_{inner_radius}_to_{outer_radius}_latent_samples'
            reconstructions = self._decoder(latent_samples)
            reconstructions_grid = torchvision.utils.make_grid(reconstructions, padding=0, normalize=True)
            self.logger.experiment.add_image(log_title, reconstructions_grid, global_step=self._train_step_counter)

    def _log_ood_reconstructions(self, set_name: str, dataloader: DataLoader) -> None:
        """Log reconstruction and residuals of batches from OOD data."""
        n_batches = 2
        for batch in islice(dataloader, n_batches):
            in_batch = batch['scan'].cuda()
            mask = batch['mask'].cuda() if 'mask' in batch.keys() else None
            rec_batch, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(in_batch,
                                                                                                               mask)
            residuals, grad_residuals = self._get_residuals_and_gradient(in_batch, rec_batch)
            grid = self._create_in_rec_res_grad_grid(in_batch, rec_batch, residuals, grad_residuals)
            self.logger.experiment.add_image(f'OOD_{set_name}', grid, global_step=self._train_step_counter)

    def _log_random_eval_batch_reconstruction(self, outputs: List[Dict],
                                              n_batches: int = 2, max_samples: int = 32) -> None:
        """Sample batches from from validation steps and visualize input, output, reconstruction and gradients."""
        np.random.seed(0)  # Make sure we always use the same samples for visualization
        out_samples = np.random.choice(outputs, min(len(outputs), n_batches), replace=False)
        for idx, out_sample in enumerate(out_samples):
            in_batch, rec_batch = out_sample['input_batch'], out_sample['reconstructed_batch']
            n_slices, _, height, width = in_batch.shape
            if max_samples < n_slices:  # Limit number of samples shown if batch-size is large to reduce storage
                np.random.seed(0)
                slice_indices = sorted(np.random.choice(list(range(n_slices)), max_samples, replace=False))
                in_batch = in_batch[slice_indices, :, :, :]
                rec_batch = rec_batch[slice_indices, :, :, :]
            residuals, grad_residuals = self._get_residuals_and_gradient(in_batch, rec_batch)
            grid = self._create_in_rec_res_grad_grid(in_batch, rec_batch, residuals, grad_residuals)
            self.logger.experiment.add_image(f'ID_random_batch_{idx + 1}', grid, global_step=self._train_step_counter)

    def _get_residuals_and_gradient(self, input_batch: Tensor, rec_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """For a given input batch and a reconstruction batch, create residual and residual gradient batches."""
        residuals = residual_l1_max(rec_batch, input_batch)
        grad_residuals = self._gradient_net.forward(residuals)
        return residuals, grad_residuals

    @staticmethod
    def _create_in_rec_res_grad_grid(in_batch: Tensor, rec_batch: Tensor,
                                     res_batch: Tensor, res_grad_batch: Tensor):
        """Create a torchvision grid vom input, reconstruction, residual and residual gradient batches."""
        input_img_grid = torchvision.utils.make_grid(in_batch, padding=0, normalize=True)
        output_img_grid = torchvision.utils.make_grid(rec_batch, padding=0, normalize=True)
        residual_img_grid = torchvision.utils.make_grid(res_batch, padding=0, normalize=True)
        grad_diff_grid = torchvision.utils.make_grid(res_grad_batch, padding=0, normalize=True)
        grid = torchvision.utils.make_grid([input_img_grid, output_img_grid, residual_img_grid, grad_diff_grid])
        return grid


def load_vae_baur_model(checkpoint_path: Path) -> nn.Module:
    assert checkpoint_path.exists(), f'Model checkpoint does not exist!'
    model = VariationalAutoEncoder(BaurEncoder(), BaurDecoder())
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

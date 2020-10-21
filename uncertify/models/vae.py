"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import dataclasses

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
import torch.distributions as dist

from uncertify.models.gradient import Gradient
from uncertify.models.beta_annealing import monotonic_annealing, cyclical_annealing, sigmoid_annealing
from uncertify.models.beta_annealing import BetaConfig, ConstantBetaConfig, MonotonicBetaConfig, \
    CyclicBetaConfig, SigmoidBetaConfig
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Dict, Callable

LATENT_SPACE_DIM = 128


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 beta_config: BetaConfig = ConstantBetaConfig(beta=1.0),
                 n_m_factor: float = 1.0) -> None:
        """Variational Auto Encoder which works with generic encoders and decoders.
        Args:
            encoder: the encoder module (can be pytorch or lightning module)
            decoder: the decoder module (can be pytorch or lightning module)
            beta_config: configuration on how to handle the beta parameter (constant, monotonic or cyclic)
        """
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._gradient_net = Gradient()
        self._train_step_counter = 0
        self._beta_config = beta_config
        self.save_hyperparameters('decoder', 'encoder')
        self._n_m_factor = n_m_factor

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            x: raw image tensor (un-flattened)
        """
        mu, log_var = self._encoder(x)
        latent_code = self._reparameterize(mu, log_var)
        reconstruction = self._decoder(latent_code)
        total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err = self.loss_function(reconstruction, x, mu, log_var)
        return reconstruction, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code

    @staticmethod
    def _reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Applying reparameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(std^2) = 2*log(std) -> std = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']

        # Run batch through model and get losses
        rec_batch, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(features)

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

        # Run validation batch through model and get losses
        rec_batch, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(features)

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

        # Sample batches from from validation steps and visualize
        np.random.seed(0)  # Make sure we always use the same samples for visualization
        out_samples = np.random.choice(outputs, min(len(outputs), 2), replace=False)
        for idx, out_sample in enumerate(out_samples):
            in_batch, rec_batch = out_sample['input_batch'], out_sample['reconstructed_batch']
            input_img_grid = torchvision.utils.make_grid(in_batch, normalize=True)
            output_img_grid = torchvision.utils.make_grid(rec_batch, normalize=True)
            residuals = torch.abs(rec_batch - in_batch)
            residual_img_grid = torchvision.utils.make_grid(residuals, normalize=True)
            grad_residuals = self._gradient_net.forward(residuals)
            grad_diff_grid = torchvision.utils.make_grid(grad_residuals, normalize=True)
            grid = torchvision.utils.make_grid([input_img_grid, output_img_grid, residual_img_grid, grad_diff_grid])
            self.logger.experiment.add_image(f'random_batch_{idx + 1}', grid, global_step=self._train_step_counter)
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param.data, global_step=self._train_step_counter)

        # Sample from latent space and check reconstructions
        n_latent_samples = 32
        with torch.no_grad():
            latent_samples = torch.normal(mean=0, std=torch.ones((n_latent_samples, LATENT_SPACE_DIM,))).cuda()
            latent_sample_reconstructions = self._decoder(latent_samples)
            latent_sample_reconstructions_grid = torchvision.utils.make_grid(latent_sample_reconstructions,
                                                                             padding=0,
                                                                             normalize=True)
            self.logger.experiment.add_image(f'random_latent_sample_reconstructions',
                                             latent_sample_reconstructions_grid, global_step=self._train_step_counter)
        loss_terms.update({'avg_val_mean_total_loss': avg_val_total_losses.mean()})
        return loss_terms

    def loss_function(self, reconstruction: Tensor, observation: Tensor, mu: Tensor, log_var: Tensor,
                      beta: float = 1.0, train_step: int = None):
        # p(x|z)
        rec_dist = dist.Normal(reconstruction, 1.0)
        log_p_x_z = rec_dist.log_prob(observation)
        log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))  # log likelihood for each sample

        # p(z)
        z_prior = dist.Normal(0.0, 1.0)

        # q(z|x)
        z_post = dist.Normal(mu, torch.sqrt(torch.exp(log_var)))

        # KL(q(z|x), p(z))
        kl_div = dist.kl_divergence(z_post, z_prior)
        kl_div = torch.mean(kl_div, dim=1)  # KL divergences as we would get them per sample in the batch
        kl_div = self._calculate_beta(self._train_step_counter) * kl_div

        # Take the mean over all batches
        mean_kl_div = torch.mean(kl_div)
        mean_log_p_x_z = torch.mean(log_p_x_z)

        variational_lower_bound = (-mean_kl_div + mean_log_p_x_z) * self._n_m_factor
        total_loss = -variational_lower_bound

        return total_loss, mean_kl_div, -mean_log_p_x_z, kl_div, -log_p_x_z

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
        else:
            raise RuntimeError(f'Beta config not supported.')

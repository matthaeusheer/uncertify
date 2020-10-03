"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import dataclasses

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional
import torchvision
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
import torch.distributions as dist

from uncertify.models.gradient import Gradient
from uncertify.models.beta_annealing import monotonic_annealing, cyclical_annealing
from uncertify.models.beta_annealing import BetaConfig, ConstantBetaConfig, MonotonicBetaConfig, CyclicBetaConfig
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Dict, Callable

LATENT_SPACE_DIM = 128


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 beta_config: BetaConfig = ConstantBetaConfig(beta=1.0)) -> None:
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
        self.val_counter = 0
        self._train_step_counter = 0
        self._beta_config = beta_config
        self.save_hyperparameters('decoder', 'encoder')

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
        std = torch.exp(0.5 * log_var)  # log_var = log(sigma^2) = 2*log(sigma) -> sigma = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']
        reconstruction, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(
            features)
        logger_losses = {'train_loss': total_loss,
                         'train_reconstruction_err': mean_rec_err,
                         'train_kld_dic': mean_kl_div}
        self.logger.experiment.add_scalars('train_losses_vs_step', logger_losses,
                                           global_step=self._train_step_counter)
        self.logger.experiment.add_scalar('beta', self._calculate_beta(self._train_step_counter),
                                          global_step=self._train_step_counter)

        self._train_step_counter += 1
        return {'loss': total_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']  # some datasets (e.g. brats) holds also 'seg' batch
        reconstruction, mu, log_var, total_loss, mean_kl_div, mean_rec_err, kl_div, rec_err, latent_code = self(
            features)
        total_loss, mean_kl_div, mean_rec_err, kl_div, log_p_x_z = self.loss_function(reconstruction, features, mu,
                                                                                      log_var)

        return {'val_mean_total_loss': total_loss, 'val_mean_kl_div': mean_kl_div, 'val_mean_rec_err': mean_rec_err,
                'reconstructed_batch': reconstruction, 'input_batch': features}

    def validation_epoch_end(self, outputs: List[Dict]) -> dict:
        # Compute average validation loss
        avg_val_losses = torch.stack([x['val_mean_total_loss'] for x in outputs])
        avg_val_kld_losses = torch.stack([x['val_mean_kl_div'] for x in outputs])
        avg_val_recon_losses = torch.stack([x['val_mean_rec_err'] for x in outputs])
        losses = {'avg_val_mean_total_loss': avg_val_losses.mean(),
                  'avg_val_mean_kl_div': avg_val_kld_losses.mean(),
                  'avg_val_mean_rec_err': avg_val_recon_losses.mean()}
        self.logger.experiment.add_scalars('avg_val_mean_losses', losses, global_step=self._train_step_counter)
        # Sample batches from from validation steps and visualize
        np.random.seed(0)  # Make sure we always use the same samples for visualization
        out_samples = np.random.choice(outputs, min(len(outputs), 4), replace=False)
        for idx, out_sample in enumerate(out_samples):
            input_img_grid = torchvision.utils.make_grid(out_sample['input_batch'], normalize=True)
            output_img_grid = torchvision.utils.make_grid(out_sample['reconstructed_batch'], normalize=True)
            residuals = torch.abs(out_sample['reconstructed_batch'] - out_sample['input_batch'])
            residual_img_grid = torchvision.utils.make_grid(residuals, normalize=True)
            grad_residuals = self._gradient_net.forward(residuals)
            grad_diff_grid = torchvision.utils.make_grid(grad_residuals, normalize=True)
            grid = torchvision.utils.make_grid([input_img_grid, output_img_grid, residual_img_grid, grad_diff_grid])
            self.logger.experiment.add_image(f'random_batch_{idx + 1}', grid, global_step=self.val_counter)
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param.data, global_step=self.val_counter)

        # Sample from latent space and check reconstructions
        n_latent_samples = 32
        with torch.no_grad():
            latent_samples = torch.normal(mean=0, std=torch.ones((n_latent_samples, LATENT_SPACE_DIM,))).cuda()
            latent_sample_reconstructions = self._decoder(latent_samples)
            latent_sample_reconstructions_grid = torchvision.utils.make_grid(latent_sample_reconstructions, padding=0)
            self.logger.experiment.add_image(f'random_latent_sample_reconstructions',
                                             latent_sample_reconstructions_grid, global_step=self.val_counter)
        self.val_counter += 1
        return dict()

    def loss_function(self, reconstruction: Tensor, observation: Tensor, mu: Tensor, log_std: Tensor,
                      beta: float = 1.0, train_step: int = None):
        # p(x|z)
        rec_dist = dist.Normal(reconstruction, 1.0)
        log_p_x_z = rec_dist.log_prob(observation)
        log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))

        # p(z)
        z_prior = dist.Normal(0.0, 1.0)

        # q(z|x)
        z_post = dist.Normal(mu, torch.exp(log_std))

        # KL(q(z|x), p(z))
        kl_div = dist.kl_divergence(z_post, z_prior)
        kl_div = torch.mean(kl_div, dim=1)

        kl_div = self._calculate_beta(self._train_step_counter) * kl_div

        # Take the mean over all batches
        variational_lower_bound = -kl_div + log_p_x_z
        total_loss = torch.mean(-variational_lower_bound)
        mean_kl_div = torch.mean(kl_div)
        mean_log_p_x_z = torch.mean(log_p_x_z)

        return total_loss, mean_kl_div, -mean_log_p_x_z, kl_div, -log_p_x_z

    def configure_optimizers(self) -> Optimizer:
        """Pytorch-lightning function."""
        return torch.optim.Adam(self.parameters(), lr=2 * 1e-4)

    def _calculate_beta(self, train_step: int) -> float:
        """Calculates the KL term for any strategy for a given training_step"""
        if isinstance(self._beta_config, ConstantBetaConfig):
            return self._beta_config.beta
        elif isinstance(self._beta_config, MonotonicBetaConfig):
            return monotonic_annealing(train_step, **dataclasses.asdict(self._beta_config))
        elif isinstance(self._beta_config, CyclicBetaConfig):
            return cyclical_annealing(train_step, **dataclasses.asdict(self._beta_config))

"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional
import torchvision
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer
import torch.distributions as dist

from uncertify.models.gradient import Gradient
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Dict, Callable

LATENT_SPACE_DIM = 128


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, get_batch_fn: Callable = lambda x: x) -> None:
        """Variational Auto Encoder which works with generic encoders and decoders.
        Args:
            encoder: the encoder module (can be pytorch or lightning module)
            decoder: the decoder module (can be pytorch or lightning module)
            get_batch_fn: some dataloaders provide images differently (e.g. either image tensor directly or a dict
                          with input images and segmentation image tensors), this function takes in a batch as
                          returned by the dataloader and returns the plain input image tensor to train on
        """
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._get_batch_fn = get_batch_fn
        self._gradient_net = Gradient()
        self._train_steps = 0
        self.val_counter = 0
        self._train_step_counter = 0
        self.save_hyperparameters('decoder', 'encoder', 'get_batch_fn')

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
        features = self._get_batch_fn(batch)
        reconstructed_batch, mu, log_var, train_loss, kld_loss, reconstruction_loss = self(features)
        logger_losses = {'train_loss': train_loss,
                         'train_reconstruction_loss': reconstruction_loss,
                         'train_kld_loss': kld_loss}
        self.logger.experiment.add_scalars('train_losses_vs_step', logger_losses, global_step=self._train_step_counter)
        self._train_step_counter += 1
        return {'loss': train_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        features = self._get_batch_fn(batch)  # some datasets (e.g. brats) holds also 'seg' batch
        reconstructed_batch, mu, log_var = self(features)
        val_loss, kld_loss, reconstruction_loss = self.loss_function(reconstructed_batch, features, mu, log_var)
        return {'val_loss': val_loss, 'val_kld_loss': kld_loss, 'val_recon_loss': reconstruction_loss,
                'reconstructed_batch': reconstructed_batch, 'input_batch': features}

    def validation_epoch_end(self, outputs: List[Dict]) -> dict:
        # Compute average validation loss
        avg_val_losses = torch.stack([x['val_loss'] for x in outputs])
        avg_val_kld_losses = torch.stack([x['val_kld_loss'] for x in outputs])
        avg_val_recon_losses = torch.stack([x['val_recon_loss'] for x in outputs])
        losses = {'avg_val_loss': avg_val_losses.mean(),
                  'avg_val_kld_loss': avg_val_kld_losses.mean(),
                  'avg_val_recon_loss': avg_val_recon_losses.mean()}
        self.logger.experiment.add_scalars('avg_val_losses', losses, global_step=self._train_step_counter)

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
        self.val_counter += 1
        # Sample from latent space and check reconstructions
        n_latent_samples = 16
        latent_space_dim = 128
        latent_samples = torch.normal(mean=0, std=torch.ones((n_latent_samples, latent_space_dim, )))
        return dict()

    @staticmethod
    def old_loss_function(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper.
        Caution:
            This function returns a tuple of the individual loss terms for easy logging. Need to add them up wen used.
        """
        _, _, img_height, img_width = x_in.shape
        # reconstruction_loss = nn.BCEWithLogitsLoss(reduction='mean')(x_in, x_out)  # per pix
        reconstruction_loss = torch_functional.l1_loss(x_out, x_in) / (img_width * img_height)
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)  # mean over batches
        total_loss = reconstruction_loss + kld
        return total_loss, reconstruction_loss, kld

    @staticmethod
    def loss_function(reconstruction, observation, mu, log_std, kld_multiplier=1.0):
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
        kl_div = kld_multiplier * kl_div

        # Take the mean over all batches
        variational_lower_bound = -kl_div + log_p_x_z
        total_loss = torch.mean(-variational_lower_bound)
        mean_kl_div = torch.mean(kl_div)
        mean_log_p_x_z = torch.mean(log_p_x_z)

        return total_loss, mean_kl_div, -mean_log_p_x_z, kl_div, -log_p_x_z

    def configure_optimizers(self) -> Optimizer:
        """Pytorch-lightning function."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

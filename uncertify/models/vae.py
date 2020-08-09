"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import numpy as np
import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from uncertify.models.gradient import Gradient
from uncertify.utils.custom_types import Tensor

from typing import Tuple, List, Dict, Callable


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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            x: raw image tensor (un-flattened)
        """
        mu, log_var = self._encoder(x)
        latent_code = self._reparameterize(mu, log_var)
        reconstruction = self._decoder(latent_code)
        return reconstruction, mu, log_var

    @staticmethod
    def _reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Applying reparameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(sigma^2) = 2*log(sigma) -> sigma = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        features = self._get_batch_fn(batch)
        reconstructed_batch, mu, log_var = self(features)
        reconstruction_loss, kld_loss = self.loss_function(reconstructed_batch, features, mu, log_var)
        train_loss = reconstruction_loss + kld_loss
        logger_losses = {'train_loss': train_loss,
                         'train_reconstruction_loss': reconstruction_loss,
                         'train_kld_loss': kld_loss}
        self.logger.experiment.add_scalars('train_losses_vs_step', logger_losses, global_step=self._train_step_counter)
        self._train_step_counter += 1
        return {'loss': train_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        features = self._get_batch_fn(batch)  # some datasets (e.g. brats) holds also 'seg' batch
        reconstructed_batch, mu, log_var = self(features)
        reconstruction_loss, kld_loss = self.loss_function(reconstructed_batch, features, mu, log_var)
        val_loss = reconstruction_loss + kld_loss
        return {'val_loss': val_loss, 'val_kld_loss': kld_loss, 'val_recon_loss': reconstruction_loss,
                'reconstructed_batch': reconstructed_batch, 'input_batch': features}

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        # Compute average validation loss
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_kld_loss = torch.stack([x['val_kld_loss'] for x in outputs]).mean()
        avg_val_recon_loss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        losses = {'avg_val_loss': avg_val_loss,
                  'avg_val_kld_loss': avg_val_kld_loss,
                  'avg_val_recon_loss': avg_val_recon_loss}
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

    @staticmethod
    def loss_function(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tuple[Tensor, Tensor]:
        """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper.
        Caution:
            This function returns a tuple of the individual loss terms for easy logging. Need to add them up wen used.
        """
        kld_factor = 1  #
        reconstruction_loss = nn.BCEWithLogitsLoss(reduction='sum')(x_in, x_out)
        kld_loss = kld_factor * (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()))  # See Appendix B of paper
        return reconstruction_loss, kld_loss

    def configure_optimizers(self) -> Optimizer:
        """Pytorch-lightning function."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

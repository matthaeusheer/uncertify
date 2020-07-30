"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from uncertify.models.encoder_decoder import Encoder, Decoder
from uncertify.common import DATA_DIR_PATH
from uncertify.models.custom_types import Tensor

from typing import Tuple


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """A PyTorch VariationalAutoEncoder model using linear layers only."""
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        print(self)

    @staticmethod
    def _reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Applying reparameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(sigma^2) = 2*log(sigma) -> sigma = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            x: raw image tensor (un-flattened)
        """
        print(f'Forward - input shape {x.shape}')
        mu, log_var = self._encoder(x)
        print(f'Forward - mu, logvar shapes {mu.shape} / {log_var.shape}')
        latent_code = self._reparameterize(mu, log_var)
        print(f'Forward - latent code shape {latent_code.shape}')

        return self._decoder(latent_code), mu, log_var

    def training_step(self, batch, batch_idx):
        """Pytorch-lightning function."""
        features, _ = batch
        print(features.shape)
        reconstructed_batch, mu, log_var = self(features)
        train_loss = self.loss_function(reconstructed_batch, features.view(-1, self.input_dim), mu, log_var)
        log = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        """Pytorch-lightning function."""
        features, _ = batch
        reconstructed_batch, mu, log_var = self(features)
        val_loss = self.loss_function(reconstructed_batch, features.view(-1, self.input_dim), mu, log_var)
        return {'val_loss': val_loss, 'reconstructed_batch': reconstructed_batch}

    def loss_function(self, x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor):
        """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper."""
        # reconstruction_loss = F.mse_loss(x_out, x_in)  # TODO: Why does this not work?
        reconstruction_loss = F.binary_cross_entropy(x_in, x_out, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
        return reconstruction_loss + kld_loss

    def configure_optimizers(self):
        """Pytorch-lightning function."""
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def validation_epoch_end(self, outputs):
        """Pytorch-lightning function."""
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['reconstructed_batch'][0]
        log = {'avg_val_loss': avg_val_loss}
        grid = torchvision.utils.make_grid(x_hat.view(28, 28))  # remove hardcode
        self.logger.experiment.add_image('image', grid, 0)
        return {'log': log}

    def train_dataloader(self) -> DataLoader:
        """Pytorch-lightning function."""
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root=DATA_DIR_PATH / 'cifar10_data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
        return DataLoader(train_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self) -> DataLoader:
        """Pytorch-lightning function."""
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        val_set = torchvision.datasets.CIFAR10(root=DATA_DIR_PATH / 'cifar10_data',
                                               train=False,
                                               download=True,
                                               transform=transform)
        return DataLoader(val_set,
                          batch_size=4,
                          shuffle=False,
                          num_workers=4)


def vae_loss(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper."""
    # reconstruction_loss = F.mse_loss(x_out, x_in)  # TODO: Why does this not work?
    reconstruction_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
    return reconstruction_loss + kld_loss


if __name__ == '__main__':
    trainer_kwargs = {'max_epochs': 10}
    trainer = pl.Trainer(**trainer_kwargs, fast_dev_run=True)
    encoder = Encoder(in_channels=3, hidden_dims=[10, 10], latent_dim=2)
    decoder = Decoder(latent_dim=2, hidden_dims=[10, 10], img_shape=(32, 32), reverse_hidden_dims=True)
    model = VariationalAutoEncoder(encoder, decoder)
    trainer.fit(model)

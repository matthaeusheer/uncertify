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

from uncertify.common import DATA_DIR_PATH

from typing import Tuple, TypeVar

Tensor = TypeVar('torch.tensor')


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, bottleneck_dim: int = 50) -> None:
        """A PyTorch VariationalAutoEncoder model using linear layers only."""
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc_var = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode an input batch by passing it through the encoder network.
        Args:
            x: a tensor with shape [B x C x H x W]
        Returns:
            latent code consist of mean and log variance of latent prior distribution
        """
        hidden_1 = F.relu(self.fc1(x))
        mu, log_var = self.fc_mu(hidden_1), self.fc_var(hidden_1)
        return mu, log_var

    def _decode(self, latent_vector: Tensor):
        """Decode a latent code into a reconstruction.
        Args:
            latent_vector: latent code represented by a mean (mu) and log variance (log_var)
        Returns:
            a reconstruction???
        """
        hidden_3 = F.relu(self.fc3(latent_vector))
        return F.sigmoid(self.fc4(hidden_3))

    @staticmethod
    def _reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Applying reparameterization trick."""
        std = torch.exp(0.5 * log_var)  # log_var = log(sigma^2) = 2*log(sigma) -> sigma = exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _flatten(self, x: Tensor) -> Tensor:
        """Flattens a Tensor in image format (e.g. 28x28) to a flat vector in input format dimensions (e.g. 784)."""
        return x.view(-1, self.input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            x: raw image tensor (un-flattened)
        """
        mu, log_var = self._encode(self._flatten(x))
        latent_vector = VariationalAutoEncoder._reparameterize(mu, log_var)
        return self._decode(latent_vector), mu, log_var

    def training_step(self, batch, batch_idx):
        """Pytorch-lightning function."""
        features, _ = batch
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
        grid = torchvision.utils.make_grid(x_hat.view(28, 28))
        self.logger.experiment.add_image('image', grid, 0)
        return {'log': log}

    def train_dataloader(self) -> DataLoader:
        """Pytorch-lightning function."""
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
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
        val_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                             train=False,
                                             download=True,
                                             transform=transform)
        return DataLoader(val_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4)


def vae_loss(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper."""
    # reconstruction_loss = F.mse_loss(x_out, x_in)  # TODO: Why does this not work?
    reconstruction_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
    return reconstruction_loss + kld_loss


if __name__ == '__main__':
    trainer_kwargs = {'max_epochs': 10}
    trainer = pl.Trainer(**trainer_kwargs, fast_dev_run=False)
    model = VariationalAutoEncoder()
    trainer.fit(model)

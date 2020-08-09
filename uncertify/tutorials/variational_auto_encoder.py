"""
Example of a simple VariationalAutoEncoder implementation in plain PyTorch.

NOTE: This is a raw Pytorch implementation. A Pytorch Lightning module can be found in the models folder while
      this implementation here is only for tutorial / learning purposes.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt

from uncertify.common import DATA_DIR_PATH
from uncertify.utils.custom_types import Tensor

from typing import Tuple

from uncertify.visualization.reconstruction import plot_vae_reconstructions, plot_vae_generations


class VariationalAutoEncoder(torch.nn.Module):
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feed forward computation.
        Args:
            x: raw image tensor (un-flattened)
        """
        mu, log_var = self._encode(self._flatten(x))
        latent_vector = VariationalAutoEncoder._reparameterize(mu, log_var)
        return self._decode(latent_vector), mu, log_var

    def _flatten(self, x: Tensor) -> Tensor:
        """Flattens a Tensor in image format (e.g. 28x28) to a flat vector in input format dimensions (e.g. 784)."""
        return x.view(-1, self.input_dim)


def vae_loss(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper."""
    # reconstruction_loss = F.mse_loss(x_out, x_in)  # TODO: Why does this not work?
    reconstruction_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
    return reconstruction_loss + kld_loss


def train_vae(model: VariationalAutoEncoder, device: torch.device, train_loader: DataLoader, test_loader: DataLoader,
              optimizer: Optimizer, n_epochs: int, val_frequency: int,
              sampled_z: Tensor = None) -> VariationalAutoEncoder:
    """Main training loop for the VariationalAutoEncoder. After each epoch there is one full validation pass."""
    model = model.to(device)
    fig_counter = 0
    for epoch_idx in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (batch_features, _) in enumerate(train_loader):
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            batch_hat, mu, log_var = model.forward(batch_features)
            loss = vae_loss(batch_features.view(-1, 784), batch_hat, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % (len(train_loader) // 5) == 0:
                if sampled_z is not None:
                    samples = model._decode(sampled_z).view(-1, 28, 28).cpu().detach().numpy()
                    fig = plot_vae_generations(samples)
                    fig.savefig(str(DATA_DIR_PATH / 'vae_movie' / f'sampled_{fig_counter}.png'))
                reconstruct_fig = plot_vae_reconstructions(model, test_loader, device, n_batches=1, max_samples=1,
                                                           show=False)[0]
                reconstruct_fig.savefig(str(DATA_DIR_PATH / 'vae_movie' / f'reconstruct_{fig_counter}.png'))
                fig_counter += 1
                print(f'Train Epoch: {epoch_idx + 1} [{batch_idx * len(batch_features)}/{len(train_loader.dataset)}] '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(batch_features):.6f}')
        print(f'==> Epoch {epoch_idx + 1} average loss: {epoch_loss / len(train_loader.dataset)}')
        if (epoch_idx + 1) % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_idx, (batch_features, _) in enumerate(test_loader):
                    batch_features = batch_features.to(device)
                    batch_hat, mu, log_var = model(batch_features)
                    val_loss += vae_loss(batch_features.view(-1, 784), batch_hat, mu, log_var).item()
                val_loss /= len(test_loader.dataset)
                print(f'==> Epoch {epoch_idx + 1} Validation loss: {val_loss:.3f}')
        plt.close('all')  # free all matplotlib figures
    return model



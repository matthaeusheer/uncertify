"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

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
        mu, log_var = self._encoder(x)
        latent_code = self._reparameterize(mu, log_var)
        reconstruction = self._decoder(latent_code)
        return reconstruction, mu, log_var

    def training_step(self, batch, batch_idx):
        """Pytorch-lightning function."""
        features, _ = batch
        reconstructed_batch, mu, log_var = self(features)
        train_loss = self.loss_function(reconstructed_batch, features, mu, log_var)
        log = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        """Pytorch-lightning function."""
        features, _ = batch
        reconstructed_batch, mu, log_var = self(features)
        val_loss = self.loss_function(reconstructed_batch, features, mu, log_var)
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
        grid = torchvision.utils.make_grid(x_hat)  # remove hardcode
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
                          num_workers=0)

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
                          num_workers=0)


def vae_loss(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """Loss function of Variational AutoEncoder as stated in original 'Autoencoding Variational Bayes' paper."""
    reconstruction_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
    return reconstruction_loss + kld_loss


if __name__ == '__main__':
    latent_dims = 100
    hidden_conv_dims = [32, 64, 128, 265, 512]
    input_channels = 3  # depends on dataset
    flat_conv_output_dim = 512  # run and check the dimension for the settings above

    trainer_kwargs = {'max_epochs': 10, 'gpus': 1}
    trainer = pl.Trainer(**trainer_kwargs, fast_dev_run=False)
    encoder_net = Encoder(in_channels=input_channels, hidden_dims=hidden_conv_dims,
                          latent_dim=latent_dims, flat_conv_output_dim=flat_conv_output_dim)
    decoder_net = Decoder(latent_dim=latent_dims, hidden_dims=hidden_conv_dims,
                          flat_conv_output_dim=flat_conv_output_dim, reverse_hidden_dims=True)
    model = VariationalAutoEncoder(encoder_net, decoder_net)
    trainer.fit(model)

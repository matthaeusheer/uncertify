"""
A Variational AutoEncoder model implemented using pytorch lightning.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from uncertify.models.encoder_decoder import Encoder, Decoder
from uncertify.common import DATA_DIR_PATH
from uncertify.models.gradient import Gradient
from uncertify.models.custom_types import Tensor

from typing import Tuple, List, Dict


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        """Variational Auto Encoder which works with generic encoders and decoders.

        Note that the tensor dimension handling is completely handled by the provided encoder and decoder.
        See the "forward" method on how those are used within this class.
        """
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._gradient_net = Gradient()
        self.val_counter = 0

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
        return {'val_loss': val_loss, 'reconstructed_batch': reconstructed_batch, 'input_batch': features}

    @staticmethod
    def loss_function(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor):
        """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper."""
        reconstruction_loss = F.binary_cross_entropy(x_in, x_out, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # See Appendix B of paper
        return reconstruction_loss + kld_loss

    def configure_optimizers(self):
        """Pytorch-lightning function."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_epoch_end(self, outputs: List[Dict]):
        # Compute average validation loss
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'avg_val_loss': avg_val_loss}

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
        self.val_counter += 1
        return {'log': log}

    def train_dataloader(self) -> DataLoader:
        """Pytorch-lightning function."""
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                    torchvision.transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                               train=True,
                                               download=True,
                                               transform=transform)
        return DataLoader(train_set,
                          batch_size=8,
                          shuffle=True,
                          num_workers=0)

    def val_dataloader(self) -> DataLoader:
        """Pytorch-lightning function."""
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                    torchvision.transforms.ToTensor()])
        val_set = torchvision.datasets.MNIST(root=DATA_DIR_PATH / 'mnist_data',
                                             train=False,
                                             download=True,
                                             transform=transform)
        return DataLoader(val_set,
                          batch_size=8,
                          shuffle=False,
                          num_workers=0)


if __name__ == '__main__':
    latent_dims = 100
    hidden_conv_dims = [32, 64, 128, 265, 512]
    input_channels = 1  # depends on dataset, greyscale: 1, RGB: 3
    flat_conv_output_dim = 512  # run and check the dimension for the settings above

    logger = TensorBoardLogger(str(DATA_DIR_PATH / 'lightning_logs'), name='vae_test')

    trainer_kwargs = {'logger': logger,
                      'max_epochs': 20,
                      'val_check_interval': 0.25,  # check (1 / value) * times per train epoch
                      'gpus': 1,
                      # 'precision': 16,  # needs apex installed
                      # 'train_percent_check': 100,
                      'fast_dev_run': False}
    trainer = pl.Trainer(**trainer_kwargs)
    encoder_net = Encoder(in_channels=input_channels, hidden_dims=hidden_conv_dims,
                          latent_dim=latent_dims, flat_conv_output_dim=flat_conv_output_dim)
    decoder_net = Decoder(latent_dim=latent_dims, hidden_dims=hidden_conv_dims,
                          flat_conv_output_dim=flat_conv_output_dim, output_channels=input_channels,
                          reverse_hidden_dims=True)
    model = VariationalAutoEncoder(encoder_net, decoder_net)
    trainer.fit(model)

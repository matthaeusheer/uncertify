import math

import torch
import torch.nn.functional as torch_functional
import torch.distributions as dist

from uncertify.models.vae import VariationalAutoEncoder
from uncertify.utils.custom_types import Tensor

from typing import Tuple


class SimpleVariationalAutoEncoder(VariationalAutoEncoder):
    @staticmethod
    def loss_function(x_in: Tensor, x_out: Tensor, mu: Tensor, log_var: Tensor,
                      beta: float = 1.0, train_step: int = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Loss function of Variational Autoencoder as stated in original 'Autoencoding Variational Bayes' paper.
        Caution:
            This function returns a tuple of the individual loss terms for easy logging. Need to add them up wen used.
        """
        batch_size, _, img_height, img_width = x_in.shape

        # Reconstruction error per sample for a batch
        reconstruction_loss = torch.sum(torch_functional.l1_loss(x_in, x_out, reduction='none'), dim=(1, 2, 3))

        # var = sigma**2
        kld = 0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)

        total_loss = torch.mean(-kld + reconstruction_loss)
        return total_loss, torch.mean(reconstruction_loss), torch.mean(kld)

    def forward(self, img_tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self._encoder(img_tensor)
        latent_code = self._reparameterize(mu, log_var)
        reconstruction = self._decoder(latent_code)
        total_loss, rec_loss, kld = self.loss_function(reconstruction, img_tensor, mu, log_var)
        return reconstruction, mu, log_var, total_loss, rec_loss, kld, latent_code

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']
        reconstruction, mu, log_var, total_loss, rec_loss, kld, latent_code = self(features)
        logger_losses = {'train_loss': total_loss,
                         'train_reconstruction_err': rec_loss,
                         'train_kl_div': kld}
        self.logger.experiment.add_scalars('train_losses_vs_step', logger_losses, global_step=self._train_step_counter)
        self._train_step_counter += 1
        return {'loss': total_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        features = batch['scan']  # some datasets (e.g. brats) holds also 'seg' batch
        reconstruction, mu, log_var, total_loss, rec_loss, kld, latent_code = self(features)
        total_loss, rec_loss, kl_div = self.loss_function(reconstruction, features, mu, log_var)

        # Those values will be tracked and can be accessed in validation_epoch_end
        val_return_dict = {'val_total_loss': total_loss,
                           'val_kl_div': kl_div,
                           'val_rec_err': rec_loss,
                           'reconstructed_batch': reconstruction,
                           'input_batch': features}
        return val_return_dict

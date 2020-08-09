import torch
from torch import nn
import pytorch_lightning as pl

from uncertify.utils.custom_types import Tensor

from typing import List, Tuple

# Default values work for 32x32 Greyscale input images!
DEFAULT_HIDDEN_DIMS = [32, 64, 128, 256, 512]
DEFAULT_LATENT_DIMS = 100
DEFAULT_CHANNELS = 1
DEFAULT_FLAT_CONV_OUTPUT_DIMS = 512
DEFAULT_CONV_KERNEL_SIZE = 3
DEFAULT_CONV_STRIDE = 2
DEFAULT_CONV_PADDING = 1


class Encoder(pl.LightningModule):
    def __init__(self, in_channels: int = DEFAULT_CHANNELS, hidden_dims: List[int] = None,
                 latent_dim: int = DEFAULT_LATENT_DIMS,
                 flat_conv_output_dim: int = DEFAULT_FLAT_CONV_OUTPUT_DIMS) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS
        modules = []
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=hidden_dim,
                              kernel_size=DEFAULT_CONV_KERNEL_SIZE,
                              stride=DEFAULT_CONV_STRIDE,
                              padding=DEFAULT_CONV_PADDING),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = hidden_dim
        self._encoder = nn.Sequential(*modules)

        self._fully_connected_mu = nn.Linear(flat_conv_output_dim, latent_dim)  # from 2D shape to single channel
        self._fully_connected_var = nn.Linear(flat_conv_output_dim, latent_dim)

    def _encode(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Passing the input image batch through the encoder network and returns the latent code.
        Args:
            input_tensor: tensor batch of shape [batch_size x n_channels x img_height x img_width]
        Returns:
            a tuple of tensors representing the latent code in form of mean and variance
        """
        result = self._encoder(input_tensor)
        result = torch.flatten(result, start_dim=1)  # Flatten batches of 2D images into batches of 1D arrays
        mu, var = self._fully_connected_mu(result), self._fully_connected_var(result)
        return mu, var

    def forward(self, input_tensor: Tensor):
        return self._encode(input_tensor)


class Decoder(pl.LightningModule):
    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIMS, hidden_dims: List[int] = None,
                 flat_conv_output_dim: int = DEFAULT_FLAT_CONV_OUTPUT_DIMS, output_channels: int = DEFAULT_CHANNELS,
                 reverse_hidden_dims: bool = True) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = DEFAULT_HIDDEN_DIMS
        if reverse_hidden_dims:
            hidden_dims.reverse()
        self._hidden_dims = hidden_dims
        self._flat_conv_outdim_dim = flat_conv_output_dim

        self._decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)  # go back to 2D shape
        modules = []
        for layer_idx in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[layer_idx],
                                       hidden_dims[layer_idx + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[layer_idx + 1]),
                    nn.LeakyReLU())
            )

        self._decoder = nn.Sequential(*modules)
        self._final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=output_channels,
                      kernel_size=3,
                      padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def _decode(self, latent_code: Tensor) -> Tensor:
        """Map latent code back into image space, i.e. perform reconstruction by generating image given some code.
        Args:
            latent_code: latent code tensor in shape [batch_size x latent_dim]
        Returns:
            a tensor in shape [batch_size x n_channels x img_height x img_width]
        """
        result = self._decoder_input(latent_code)
        result = result.view(-1, self._hidden_dims[0], 2, 2)
        result = self._decoder(result)
        result = self._final_layer(result)
        return result

    def forward(self, latent_code: Tensor):
        return self._decode(latent_code)

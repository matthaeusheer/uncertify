"""
CNN-VAE based on architecture shown in Baur et al. 2020
"""

import torch
from torch import nn
import pytorch_lightning as pl

from uncertify.utils.custom_types import Tensor

from typing import Optional


class PassThrough(nn.Module):
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x


class GenericConv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 conv_module: nn.Module = nn.Conv2d, conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU, activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d, normalization_kwargs: dict = None) -> None:
        """Generic convolution block for either standard or transpose convolution.

        Allows to have a sequential module block consisting of convolution, normalization and activation with
        custom parameters for every step.

        Args:
            conv_module: torch module representing the convolutional operation, e.g. nn.Conv2D or nn.ConvTranspose2D
            in_channels: number of input channels for convolution
            out_channels: number of output channels for convolution
            conv_further_kwargs: further keyword arguments passed to the convolution module
            activation_module: torch module representing the activation operation, e.g. nn.LeakyRelu or nn.Sigmoid
            activation_kwargs: keyword arguments passed to the activation module
            normalization_module: torch module for layer normalization, e.g. nn.BatchNorm2D
            normalization_kwargs: keyword arguments passed to the normalization module
        """
        super().__init__()
        self.conv_module_ = None
        if conv_module is not None:
            self.conv_module_ = conv_module(in_channels=in_channels, out_channels=out_channels,
                                            **(conv_further_kwargs or {}))
        self.normalization_module_ = None
        if normalization_module is not None:
            self.normalization_module_ = normalization_module(out_channels, **(normalization_kwargs or {}))
        self.activation_module_ = None
        if activation_module is not None:
            self.activation_module_ = activation_module(**(activation_kwargs or {}))

    def forward(self, tensor: Tensor) -> Tensor:
        if self.conv_module_ is not None:
            tensor = self.conv_module_(tensor)
        if self.normalization_module_ is not None:
            tensor = self.normalization_module_(tensor)
        if self.activation_module_ is not None:
            tensor = self.activation_module_(tensor)
        return tensor


class Conv2DBlock(GenericConv2DBlock):
    def __init__(self, in_channels: int, out_channels: int, conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU, activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d, normalization_kwargs: dict = None) -> None:
        super().__init__(in_channels, out_channels, nn.Conv2d, conv_further_kwargs, activation_module,
                         activation_kwargs, normalization_module, normalization_kwargs)


class ConvTranspose2DBlock(GenericConv2DBlock):
    def __init__(self, in_channels: int, out_channels: int, conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU, activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d, normalization_kwargs: dict = None) -> None:
        super().__init__(in_channels, out_channels, nn.ConvTranspose2d, conv_further_kwargs, activation_module,
                         activation_kwargs, normalization_module, normalization_kwargs)


class BaurEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=32,
                            conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv2 = Conv2DBlock(in_channels=32,
                            out_channels=64,
                            conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv3 = Conv2DBlock(in_channels=64,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv4 = Conv2DBlock(in_channels=128,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv5 = Conv2DBlock(in_channels=128,
                            out_channels=16,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1})
        self._conv_layers = nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])
        self._fully_connected_mu = nn.Linear(1024, 128)
        self._fully_connected_log_var = nn.Linear(1024, 128)

    def forward(self, input_tensor: Tensor) -> Tensor:
        tensor = self._conv_layers(input_tensor)
        tensor = torch.flatten(tensor, start_dim=1)
        mu = self._fully_connected_mu(tensor)
        log_var = self._fully_connected_log_var(tensor)
        return mu, log_var


class BaurDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._fully_connected = nn.Linear(128, 1024)

        conv1 = ConvTranspose2DBlock(in_channels=16,
                                     out_channels=128,
                                     conv_further_kwargs={'kernel_size': 1, 'stride': 1})
        conv2 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=128,
                                     conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv3 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=64,
                                     conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv4 = ConvTranspose2DBlock(in_channels=64,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv5 = ConvTranspose2DBlock(in_channels=32,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 2, 'stride': 2})
        conv6 = Conv2DBlock(in_channels=32,
                            out_channels=1,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            normalization_module=None,
                            activation_module=None)
        self._conv_layers = nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])

    def forward(self, latent_code: Tensor) -> Tensor:
        flat_tensor = self._fully_connected(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        tensor = self._conv_layers(reshaped_tensor)
        return tensor

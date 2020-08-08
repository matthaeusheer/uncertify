"""
CNN-VAE based on architecture shown in Baur et al. 2020
"""

import torch
from torch import nn
import pytorch_lightning as pl

from uncertify.models.custom_types import Tensor


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self._block(tensor)


class TransposeConv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self._block(tensor)


class BaurEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=32,
                            kernel_size=2,
                            stride=2)
        conv2 = Conv2DBlock(in_channels=32,
                            out_channels=64,
                            kernel_size=2,
                            stride=2)
        conv3 = Conv2DBlock(in_channels=64,
                            out_channels=128,
                            kernel_size=2,
                            stride=2)
        conv4 = Conv2DBlock(in_channels=128,
                            out_channels=128,
                            kernel_size=2,
                            stride=2)
        conv5 = Conv2DBlock(in_channels=128,
                            out_channels=16,
                            kernel_size=1,
                            stride=1)
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

        conv1 = Conv2DBlock(in_channels=16,
                            out_channels=128,
                            kernel_size=1,
                            stride=1)
        conv2 = TransposeConv2DBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=2,
                                     stride=2)
        conv3 = TransposeConv2DBlock(in_channels=128,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=2)
        conv4 = TransposeConv2DBlock(in_channels=64,
                                     out_channels=32,
                                     kernel_size=2,
                                     stride=2)
        conv5 = TransposeConv2DBlock(in_channels=32,
                                     out_channels=32,
                                     kernel_size=2,
                                     stride=2)
        conv6 = Conv2DBlock(in_channels=32,
                            out_channels=1,
                            kernel_size=1,
                            stride=1)
        self._conv_layers = nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])

    def forward(self, latent_code: Tensor) -> Tensor:
        flat_tensor = self._fully_connected(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        tensor = self._conv_layers(reshaped_tensor)
        return tensor

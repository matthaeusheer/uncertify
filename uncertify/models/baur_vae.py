"""
CNN-VAE based on architecture shown in Baur et al.
"""

import torch
from torch import nn
import pytorch_lightning as pl

from uncertify.models.custom_types import Tensor


class BaurEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(in_channels=1,
                                out_channels=32,
                                kernel_size=2,
                                stride=2)
        self._conv2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=2,
                                stride=2)
        self._conv3 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=2,
                                stride=2)
        self._conv4 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=2,
                                stride=2)
        self._conv5 = nn.Conv2d(in_channels=128,
                                out_channels=16,
                                kernel_size=1,
                                stride=1)
        self._fully_connected_mu = nn.Linear(1024, 128)
        self._fully_connected_log_var = nn.Linear(1024, 128)

    def forward(self, input_tensor: Tensor) -> Tensor:
        tensor = torch.relu(self._conv1(input_tensor))
        tensor = torch.relu(self._conv2(tensor))
        tensor = torch.relu(self._conv3(tensor))
        tensor = torch.relu(self._conv4(tensor))
        tensor = torch.relu(self._conv5(tensor))
        tensor = torch.flatten(tensor, start_dim=1)
        mu = self._fully_connected_mu(tensor)
        log_var = self._fully_connected_log_var(tensor)
        return mu, log_var


class BaurDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._fully_connected = nn.Linear(128, 1024)

        self._conv1 = nn.Conv2d(in_channels=16,
                                out_channels=128,
                                kernel_size=1,
                                stride=1)
        self._conv2 = nn.ConvTranspose2d(in_channels=128,
                                         out_channels=128,
                                         kernel_size=2,
                                         stride=2)
        self._conv3 = nn.ConvTranspose2d(in_channels=128,
                                         out_channels=64,
                                         kernel_size=2,
                                         stride=2)
        self._conv4 = nn.ConvTranspose2d(in_channels=64,
                                         out_channels=32,
                                         kernel_size=2,
                                         stride=2)
        self._conv5 = nn.ConvTranspose2d(in_channels=32,
                                         out_channels=32,
                                         kernel_size=2,
                                         stride=2)
        self._conv6 = nn.Conv2d(in_channels=32,
                                out_channels=1,
                                kernel_size=1,
                                stride=1)

    def forward(self, latent_code: Tensor) -> Tensor:
        flat_tensor = self._fully_connected(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        tensor = torch.relu(self._conv1(reshaped_tensor))
        tensor = torch.relu(self._conv2(tensor))
        tensor = torch.relu(self._conv3(tensor))
        tensor = torch.relu(self._conv4(tensor))
        tensor = torch.relu(self._conv5(tensor))
        tensor = torch.relu(self._conv6(tensor))
        return tensor

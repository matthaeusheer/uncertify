"""
CNN-VAE based on architecture shown in Baur et al. 2020
"""

import torch
from torch import nn
import pytorch_lightning as pl

from uncertify.models.utils import conv2d_output_shape, convtranspose2d_output_shape
from uncertify.utils.custom_types import Tensor

from typing import Optional, Tuple


class PassThrough(nn.Module):
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x


class GenericConv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 conv_module: nn.Module = nn.Conv2d, conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU, activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d, normalization_kwargs: dict = None,
                 dropout_rate: float = None) -> None:
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
        self.dropout_module_ = None
        if dropout_rate is not None:
            self.dropout_module_ = nn.Dropout2d(dropout_rate)

    def forward(self, tensor: Tensor) -> Tensor:
        if self.conv_module_ is not None:
            tensor = self.conv_module_(tensor)
        if self.normalization_module_ is not None:
            tensor = self.normalization_module_(tensor)
        if self.activation_module_ is not None:
            tensor = self.activation_module_(tensor)
        if self.dropout_module_ is not None:
            tensor = self.dropout_module_(tensor)
        return tensor


class Conv2DBlock(GenericConv2DBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU,
                 activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d,
                 normalization_kwargs: dict = None,
                 dropout_rate: float = None) -> None:
        super().__init__(in_channels, out_channels, nn.Conv2d, conv_further_kwargs, activation_module,
                         activation_kwargs, normalization_module, normalization_kwargs, dropout_rate)


class ConvTranspose2DBlock(GenericConv2DBlock):
    def __init__(self,
                 in_channels: int,
                 out_channels: int, conv_further_kwargs: dict = None,
                 activation_module: Optional[nn.Module] = nn.LeakyReLU,
                 activation_kwargs: dict = None,
                 normalization_module: Optional[nn.Module] = nn.BatchNorm2d,
                 normalization_kwargs: dict = None,
                 dropout_rate: float = None) -> None:
        super().__init__(in_channels, out_channels, nn.ConvTranspose2d, conv_further_kwargs, activation_module,
                         activation_kwargs, normalization_module, normalization_kwargs, dropout_rate)


class BaurEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._conv_layers = self.define_conv_layers()
        self._fully_connected_mu, self._fully_connected_log_var = self.define_fc_layers()

    @staticmethod
    def define_fc_layers() -> Tuple[nn.Linear, nn.Linear]:
        fully_connected_mu = nn.Linear(1024, 128)
        fully_connected_log_var = nn.Linear(1024, 128)
        return fully_connected_mu, fully_connected_log_var

    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        dropout_rate = None
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=32,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv2 = Conv2DBlock(in_channels=32,
                            out_channels=64,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv3 = Conv2DBlock(in_channels=64,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv4 = Conv2DBlock(in_channels=128,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=dropout_rate)
        conv5 = Conv2DBlock(in_channels=128,
                            out_channels=16,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            dropout_rate=dropout_rate)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])

    def forward(self, input_tensor: Tensor) -> Tensor:
        tensor = self._conv_layers(input_tensor)
        tensor = torch.flatten(tensor, start_dim=1)
        mu = self._fully_connected_mu(tensor)
        log_var = self._fully_connected_log_var(tensor)
        return mu, log_var


class BaurDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._fc_layer = self.define_fc_layer()
        self._conv_layers = self.define_conv_layers()

    @staticmethod
    def define_fc_layer() -> nn.Linear:
        return nn.Linear(128, 1024)

    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        dropout_rate = None
        conv1 = Conv2DBlock(in_channels=16,
                            out_channels=128,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            dropout_rate=dropout_rate)
        conv2 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=128,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv3 = ConvTranspose2DBlock(in_channels=128,
                                     out_channels=64,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv4 = ConvTranspose2DBlock(in_channels=64,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv5 = ConvTranspose2DBlock(in_channels=32,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv6 = Conv2DBlock(in_channels=32,
                            out_channels=1,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            normalization_module=None,
                            activation_module=None,
                            dropout_rate=dropout_rate)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])

    def forward(self, latent_code: Tensor) -> Tensor:
        flat_tensor = self._fc_layer(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        tensor = self._conv_layers(reshaped_tensor)
        return tensor


class SmallBaurDecoder(BaurDecoder):
    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        dropout_rate = None
        conv1 = Conv2DBlock(in_channels=16,
                            out_channels=64,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            dropout_rate=dropout_rate)
        conv2 = ConvTranspose2DBlock(in_channels=64,
                                     out_channels=32,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv3 = ConvTranspose2DBlock(in_channels=32,
                                     out_channels=16,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv4 = ConvTranspose2DBlock(in_channels=16,
                                     out_channels=8,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv5 = ConvTranspose2DBlock(in_channels=8,
                                     out_channels=4,
                                     conv_further_kwargs={'kernel_size': 5, 'stride': 2,
                                                          'padding': 2, 'output_padding': 1},
                                     dropout_rate=dropout_rate)
        conv6 = Conv2DBlock(in_channels=4,
                            out_channels=1,
                            conv_further_kwargs={'kernel_size': 1, 'stride': 1},
                            normalization_module=None,
                            activation_module=None,
                            dropout_rate=dropout_rate)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])


def print_feature_map_sizes_baur() -> None:
    """Print the feature map sizes of Baur Encoder and Decoder."""
    # Encoder
    height_width = (128, 128)
    print(f'Initial size -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 5, 'stride': 2, 'padding': 2},
        {'kernel_size': 5, 'stride': 2, 'padding': 2},
        {'kernel_size': 5, 'stride': 2, 'padding': 2},
        {'kernel_size': 5, 'stride': 2, 'padding': 2},
        {'kernel_size': 1, 'stride': 1}
    ]
    for idx, kwargs in enumerate(conv_module_kwargs):
        height_width = conv2d_output_shape(height_width, kwargs['kernel_size'],
                                           kwargs['stride'], kwargs.get('padding', 0))
        print(f'After conv module {idx + 1} ({kwargs}) -> {height_width}')

    # Decoder
    height_width = (8, 8)
    print(f'Initial size (after reshaping connected layer output) -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 1, 'stride': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 1, 'stride': 1}
    ]
    for idx, kwargs in enumerate(conv_module_kwargs):
        height_width = convtranspose2d_output_shape(height_width, kwargs['kernel_size'],
                                                    kwargs['stride'], kwargs.get('padding', 0),
                                                    1, kwargs.get('output_padding', 0))
        print(f'After transpose conv module {idx + 1} ({kwargs}) -> {height_width}')


def print_feature_map_sizes_xiaoran() -> None:
    """Print the feature map sizes of Xiaoran Encoder and Decoder."""
    # Encoder
    height_width = (128, 128)
    print(f'Initial size -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 3, 'stride': 1, 'padding': 0},
        {'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'kernel_size': 3, 'stride': 2, 'padding': 0},
    ]
    for idx, kwargs in enumerate(conv_module_kwargs):
        height_width = conv2d_output_shape(height_width, kwargs['kernel_size'],
                                           kwargs['stride'], kwargs.get('padding', 0))
        print(f'After conv module {idx + 1} ({kwargs}) -> {height_width}')

    """
    # Decoder
    height_width = (8, 8)
    print(f'Initial size (after reshaping connected layer output) -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 1, 'stride': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1},
        {'kernel_size': 1, 'stride': 1}
    ]
    for idx, kwargs in enumerate(conv_module_kwargs):
        height_width = convtranspose2d_output_shape(height_width, kwargs['kernel_size'],
                                                    kwargs['stride'], kwargs.get('padding', 0),
                                                    1, kwargs.get('output_padding', 0))
        print(f'After transpose conv module {idx + 1} ({kwargs}) -> {height_width}')

    """

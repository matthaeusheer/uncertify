from typing import Optional

from torch import nn

from uncertify.utils.custom_types import Tensor


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
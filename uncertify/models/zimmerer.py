import pytorch_lightning as pl
import torch
import torch.nn as nn

from uncertify.models.encoders_decoders import Conv2DBlock, ConvTranspose2DBlock
from uncertify.models.utils import conv2d_output_shape, convtranspose2d_output_shape
from uncertify.utils.custom_types import Tensor

from typing import Tuple


class ZimmererEncoder(pl.LightningModule):
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
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=16,
                            conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv2 = Conv2DBlock(in_channels=16,
                            out_channels=64,
                            conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv3 = Conv2DBlock(in_channels=64,
                            out_channels=256,
                            conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv4 = Conv2DBlock(in_channels=256,
                            out_channels=1024,
                            conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        return nn.Sequential(*[conv1, conv2, conv3, conv4])


class ZimmererDecoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._fc_layer = self.define_fc_layer()
        self._conv_layers = self.define_conv_layers()

    @staticmethod
    def define_fc_layer() -> nn.Linear:
        return nn.Linear(128, 1024)

    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        conv1 = ConvTranspose2DBlock(in_channels=1024,
                                     out_channels=256,
                                     conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv2 = ConvTranspose2DBlock(in_channels=256,
                                     out_channels=64,
                                     conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv3 = ConvTranspose2DBlock(in_channels=64,
                                     out_channels=16,
                                     conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv4 = ConvTranspose2DBlock(in_channels=16,
                                     out_channels=16,
                                     conv_further_kwargs={'kernel_size': 4, 'stride': 2, 'padding': 2})
        conv5 = ConvTranspose2DBlock(in_channels=16,
                                     out_channels=1,
                                     conv_further_kwargs={'kernel_size': 4, 'stride': 1, 'padding': 2},
                                     activation_module=None)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5])

    def forward(self, latent_code: Tensor) -> Tensor:
        flat_tensor = self._fc_layer(latent_code)
        reshaped_tensor = flat_tensor.view(-1, 16, 8, 8)
        tensor = self._conv_layers(reshaped_tensor)
        return tensor


def print_feature_map_sizes_zimmerer() -> None:
    """Print the feature map sizes of Zimmerer Encoder and Decoder."""
    # Encoder
    height_width = (128, 128)
    print(f'Initial size -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 4, 'stride': 2, 'padding': 2},
        {'kernel_size': 4, 'stride': 2, 'padding': 2},
        {'kernel_size': 4, 'stride': 2, 'padding': 2},
        {'kernel_size': 4, 'stride': 2, 'padding': 2},
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

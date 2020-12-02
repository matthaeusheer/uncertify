from torch import nn

from uncertify.models.encoder_decoder_baur2020 import BaurDecoder, BaurEncoder
from uncertify.models.encoders_decoders import Conv2DBlock, ConvTranspose2DBlock
from uncertify.models.utils import conv2d_output_shape, convtranspose2d_output_shape


class BaurEncoderSmall(BaurEncoder):
    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        dropout_rate = 0.5
        conv1 = Conv2DBlock(in_channels=1,
                            out_channels=32,
                            conv_further_kwargs={'kernel_size': 5, 'stride': 2, 'padding': 2},
                            dropout_rate=0.5)
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


class BaurDecoderSmall(BaurDecoder):
    @staticmethod
    def define_conv_layers() -> nn.Sequential:
        dropout_rate = 0.5
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
                            dropout_rate=0.5)
        return nn.Sequential(*[conv1, conv2, conv3, conv4, conv5, conv6])


def print_feature_map_sizes_baur_small() -> None:
    """Print the feature map sizes of Baur Encoder and Decoder."""
    # Encoder
    height_width = (128, 128)
    print(f'Initial size -> {height_width}')
    conv_module_kwargs = [
        {'kernel_size': 5, 'stride': 3, 'padding': 1},
        {'kernel_size': 5, 'stride': 3, 'padding': 2},
        {'kernel_size': 1, 'stride': 2, 'padding': 1}
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
        {'kernel_size': 1, 'stride': 1}
    ]
    for idx, kwargs in enumerate(conv_module_kwargs):
        height_width = convtranspose2d_output_shape(height_width, kwargs['kernel_size'],
                                                    kwargs['stride'], kwargs.get('padding', 0),
                                                    1, kwargs.get('output_padding', 0))
        print(f'After transpose conv module {idx + 1} ({kwargs}) -> {height_width}')

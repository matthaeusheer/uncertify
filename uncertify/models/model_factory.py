from typing import List, Tuple

from uncertify.models.vae_adaptive_cnn import Encoder, Decoder, DEFAULT_HIDDEN_DIMS


def vanilla_encoder_decoder_factory(latent_dims: int, in_channels: int, flat_conv_output_dim: int,
                                    hidden_conv_channels: List[int] = None) -> Tuple[Encoder, Decoder]:
    """Factory method yielding 'mirrored' versions of Encoder and Decoder."""
    if hidden_conv_channels is None:
        hidden_conv_channels = DEFAULT_HIDDEN_DIMS
    encoder_net = Encoder(in_channels=in_channels, hidden_dims=hidden_conv_channels,
                          latent_dim=latent_dims, flat_conv_output_dim=flat_conv_output_dim)
    decoder_net = Decoder(latent_dim=latent_dims, hidden_dims=hidden_conv_channels,
                          flat_conv_output_dim=flat_conv_output_dim, output_channels=in_channels,
                          reverse_hidden_dims=True)
    return encoder_net, decoder_net

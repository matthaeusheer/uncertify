from uncertify.models.vae_adaptive_cnn import Decoder, Encoder
from uncertify.models.model_factory import vanilla_encoder_decoder_factory
from uncertify.models.vae import VariationalAutoEncoder


def test_encoder_setup() -> None:
    encoder = Encoder(in_channels=3, hidden_dims=[50, 10], latent_dim=10)
    decoder = Decoder(latent_dim=10, hidden_dims=[50, 10], img_shape=(100, 100), reverse_hidden_dims=True)
    vae = VariationalAutoEncoder(encoder, decoder)

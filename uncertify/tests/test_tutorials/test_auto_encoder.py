import torch

from uncertify.tutorials.auto_encoder import AutoEncoder


def test_setup():
    model = AutoEncoder(input_dim=10,
                        latent_dim=3,
                        encoder_hidden_dims=None,
                        decoder_hidden_dims=None)
    print('Encoder')
    for layer in model.encoder_layers:
        print(f'\t {layer}')
    print('Decoder')
    for layer in model.decoder_layers:
        print(f'\t {layer}')
    assert len(list(model.parameters())) == 4  # two times weight matrices and two times biases


def test_forward():
    model = AutoEncoder(input_dim=784,
                        latent_dim=10,
                        encoder_hidden_dims=None,
                        decoder_hidden_dims=None)
    net_input = torch.rand(784)
    output = model.forward(net_input)
    assert len(output) == 784

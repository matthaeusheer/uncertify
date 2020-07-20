"""
Example of a simple AutoEncoder implementation in PyTorch for MNIST digit reconstruction.
"""

import torch
from torch import nn

from typing import List


class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_hidden_dims: List[int] = None,
                 decoder_hidden_dims: List[int] = None) -> None:
        """A PyTorch AutoEncoder model using linear layers only.
        Args:
            input_dim: the dimension of the input sample
            latent_dim: dimension of the latent code
            encoder_hidden_dims: output dimensions of hidden fully connected layers in encoder network
            decoder_hidden_dims: output dimensions of hidden fully connected layers in decoder network,
                                 defaults to reversed encoder_hidden_dims
        """
        super().__init__()
        for dims in [encoder_hidden_dims, decoder_hidden_dims]:
            if dims is not None:
                assert isinstance(encoder_hidden_dims, list) and len(encoder_hidden_dims) > 0, 'Must give non-empty' \
                                                                                        'list of dimensions.'
        # Build encoder layers
        self.encoder_layers = []
        if encoder_hidden_dims is not None:
            for idx, dim in enumerate(encoder_hidden_dims):
                self.encoder_layers.append(nn.Linear(in_features=input_dim if idx == 0 else encoder_hidden_dims[idx-1],
                                                     out_features=dim))
            self.encoder_layers.append(nn.Linear(in_features=encoder_hidden_dims[-1], out_features=latent_dim))
        else:
            self.encoder_layers.append(nn.Linear(in_features=input_dim, out_features=latent_dim))
        # Build decoder layers
        if decoder_hidden_dims is None and encoder_hidden_dims is not None:
            decoder_hidden_dims = list(reversed(encoder_hidden_dims))
        self.decoder_layers = []
        if decoder_hidden_dims is not None:
            for idx, dim in enumerate(decoder_hidden_dims):
                self.decoder_layers.append(nn.Linear(in_features=latent_dim if idx == 0 else decoder_hidden_dims[idx-1],
                                                     out_features=dim))
            self.decoder_layers.append(nn.Linear(in_features=decoder_hidden_dims[-1], out_features=input_dim))
        else:
            self.decoder_layers.append(nn.Linear(in_features=latent_dim, out_features=input_dim))

        # Add layers such that Pytorch can find them
        self.model = nn.Sequential(*self.encoder_layers, *self.decoder_layers)

    def forward(self, x):
        for layer in self.encoder_layers:
            x = torch.relu(layer(x))
        # at this moment x represents the latent vector z
        for layer in self.decoder_layers:
            x = torch.relu(layer(x))
        return x

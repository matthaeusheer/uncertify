from abc import ABC, abstractmethod

from torch import nn

from uncertify.models.custom_types import Tensor

from typing import List, Any


class BaseVAE(nn.Module, ABC):
    """Abstract base class for all VAE (Variational Auto Encoder) pytorch models."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def encode(self, tensor: Tensor) -> List[Tensor]:
        """Takes the input tensor and encodes it returning the latent representation.
        Args:
            tensor: an image batch tensor of shape [batch_size * number_of_channels * height * width]
        Returns:
            a list of tensors representing the (mean and variance of the) latent space representation
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent_code: Tensor) -> Any:
        """Takes the latent code and decodes / reconstructs it.
        Args:
            latent_code: a tensor representing the reparameterized latent space representation
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def generate(self, tensor: Tensor, **kwargs) -> Tensor:
        """Reconstruct a given input image batch, pass through Auto Encoder and return reconstructions.
        Args:
            tensor: an image batch tensor of shape [batch_size * number_of_channels * height * width]
        Returns:
            a tensor representing the reconstruction of same shape as input tensor
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, tensor: Tensor) -> Tensor:
        """Reconstruct a given input image batch, pass through Auto Encoder and return reconstructions.
        Args:
            tensor: an image batch tensor of shape [batch_size * number_of_channels * height * width]
        Returns:
            a tensor representing the reconstruction of same shape as input tensor
        """
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs: dict) -> Tensor:
        pass

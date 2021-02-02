import logging
from pathlib import Path
from typing import List

import torch
from torch import nn

from uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.models.vae import VariationalAutoEncoder

LOG = logging.getLogger(__name__)


def load_ensemble_models(dir_path: Path, file_names: List[str], model_type: str = 'vae_baur') -> List[nn.Module]:
    """Load multiple models (an ensemble of models).

    Arguments:
        dir_path: the directory where the models reside
        file_names: a list of file names of every model
        model_type: string representing the type of model
    Returns:
        models: a list of actual models to run inference on
    """
    assert model_type == 'vae_baur', f'No other model is defined for loading ensemble methods yet.'
    LOG.info(f'Loading ensemble of models...')
    models = []
    for name in file_names:
        models.append(load_vae_baur_model(dir_path / name))
    return models


def load_vae_baur_model(checkpoint_path: Path) -> nn.Module:
    assert checkpoint_path.exists(), f'Model (VAE Baur) checkpoint does not exist! {checkpoint_path}'
    LOG.info(f'Loading VAE Baur model: {checkpoint_path}')
    model = VariationalAutoEncoder(BaurEncoder(), BaurDecoder())
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

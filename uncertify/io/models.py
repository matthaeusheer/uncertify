from pathlib import Path
from typing import List

from torch import nn

from uncertify.models.vae import load_vae_baur_model


def load_ensemble_models(dir_path: Path, file_names: List[str], model_type: str = 'vae_baur') -> List[nn.Module]:
    assert model_type == 'vae_baur', f'No other model is defined for loading ensemble methods yet.'
    models = []
    for name in file_names:
        models.append(load_vae_baur_model(dir_path / name))
    return models

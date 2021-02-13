import math
import logging

import torch

from uncertify.utils.custom_types import Tensor

LOG = logging.getLogger(__name__)


def get_entropy(image: Tensor, mask: Tensor) -> float:
    """Calculate the normalized Shannon entropy for a single image (residual map).

    Implements normalized entropy: https://math.stackexchange.com/questions/395121/how-entropy-scales-with-sample-size
    """
    eps = 1e-4
    n_masked_pixels = torch.sum(mask)
    masked_pixels = image[mask]
    entropy = 0.0
    for pix in masked_pixels:
        try:
            entropy -= (float(pix) * math.log2(float(pix) + eps)) / math.log2(n_masked_pixels)
        except ValueError as err:  # negative pixel values will fail in log
            LOG.exception(f'Error for pixel value {pix}.')
            raise err
    if math.isnan(entropy):
        LOG.error(f'Entropy {entropy} is nan!')
    return entropy

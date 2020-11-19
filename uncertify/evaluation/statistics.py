from enum import Enum, auto

from torch import nn
from torch.utils.data import DataLoader

from typing import List


class Statistic(Enum):
    """Slice-wise inference statistics types for OOD detection using DoSE."""
    KL_DIV = auto()
    ELBO = auto()
    REC_TERM = auto()


def yield_slice_wise_statistics(model: nn.Module, dataloder: DataLoader, statistics: List[Statistic]) -> dict:
    """Evaluate slice wise statistics and return aggregated results in a statistics-dict.
    
    Returns
        statistics_dict:
    """

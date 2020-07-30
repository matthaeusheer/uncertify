import numpy as np
from torch.utils.data import Dataset, DataLoader


# TODO(matthaeus): Create parameterizable noise dummy Dataset and DataLoader classes for testing purposes.

class NoiseDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()


class NoiseDataLoader(DataLoader):
    def __init__(self) -> None:
        super().__init__()
        pass


def create_noisy_image() -> np.ndarray:
    pass

"""
The transformers in this module assume that the input sample is of a structure like

{'scan': np.ndarray,
 'mask': np.ndarray,
 'seg': np.ndarray
}

where the 'seg' field is optional (some datasets have no segmented lesions).
"""
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

from uncertify.data.np_transforms import normalize_2d_array

from typing import Dict, Tuple

DictSample = Dict[str, np.ndarray]


class DictSampleTransform(ABC):
    @staticmethod
    def check_type_with_warning(input_: dict) -> None:
        if not isinstance(input_, dict):
            raise TypeError(f'Attempting to use a dict transform with input of type {type(input_)}. Abort.')

    @abstractmethod
    def __call__(self, sample: DictSample) -> DictSample:
        raise NotImplementedError


class MaskedNormalizeTransform(DictSampleTransform):
    """Perform masked normalization on 'scan' and 'seg' (when present) data from a dataset sample."""
    def __call__(self, sample: DictSample) -> DictSample:
        self.check_type_with_warning(sample)
        scan = sample['scan']
        mask = sample['mask']
        out_dict = {'scan': normalize_2d_array(scan, mask), 'mask': mask}
        if 'seg' not in sample.keys():
            return out_dict
        else:
            out_dict.update(seg=normalize_2d_array(sample['seg'], mask))
            return out_dict


class DictReshapeTransform(DictSampleTransform):
    """Reshapes the arrays contained in the dict sample into new shape."""
    def __init__(self, new_shape: Tuple[int, int]) -> None:
        self._new_shape = new_shape

    def __call__(self, sample: DictSample) -> DictSample:
        scan = sample['scan']
        mask = sample['mask']
        out_dict = {'scan': np.reshape(scan, self._new_shape), 'mask': np.reshape(mask, self._new_shape)}
        if 'seg' not in sample.keys():
            return out_dict
        else:
            out_dict.update(seg=np.reshape(sample['seg'], self._new_shape))
            return out_dict


class DictNumpy2PILTransform(DictSampleTransform):
    """Converts the arrays contained in the dict sample into PIL images (for torchvision)."""
    def __call__(self, sample: DictSample) -> DictSample:
        scan = sample['scan']
        mask = sample['mask']
        out_dict = {'scan': Image.fromarray(scan), 'mask': Image.fromarray(mask)}
        if 'seg' not in sample.keys():
            return out_dict
        else:
            out_dict.update(seg=Image.fromarray(sample['seg']))
            return out_dict

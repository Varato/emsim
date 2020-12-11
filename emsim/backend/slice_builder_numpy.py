from typing import Union, Tuple, List, Optional
import numpy as np
from numpy.fft import rfft2, irfft2
import logging

from .slice_builder_base import OneSliceBuilderBase, MultiSlicesBuilderBase

logger = logging.getLogger(__name__)

def symmetric_bandlimit(arr):
    # arr (n0, n1, n2)
    n1, n2 = arr.shape[-2:]
    r = min(n1, n2) // 2
    kx, ky = np.meshgrid(np.arange(-n1//2, -n1//2 + n1), np.arange(-n2//2, -n2//2 + n2))
    k2 = kx**2 + ky**2
    fil = np.where(k2 <= r**2, 1, 0)
    fil = np.fft.ifftshift(fil)  # (n1, n2)
    return np.fft.ifft2(np.fft.fft2(arr, axes=(-2,-1)) * fil) # (n0, n1, n2)


class OneSliceBuilder(OneSliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 d1: float, d2: float):
        logger.debug("using numpy OneSliceBuilder")
        super(OneSliceBuilder, self).__init__(unique_elements, n1, n2, pixel_size)

    def make_one_slice(self, atom_histograms_one_slice: np.ndarray, symmetric_bandlimit: bool = True):
        # atom_histograms_one_slice (n_elems, n1, n2)
        location_phase = rfft2(atom_histograms_one_slice, axes=(-2, -1))  # (n_elems, n1, n2//2+1)
        location_phase *= self.scattering_factors                         # (n_elems, n1, n2//2+1)
        aslice = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        if symmetric_bandlimit:
            aslice = symmetric_bandlimit(aslice)
        np.clip(aslice, a_min=1e-7, a_max=None, out=aslice)
        return aslice



class MultiSlicesBuilder(MultiSlicesBuilderBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        logger.debug("using numpy MultiSlicesBuilder")
        super(MultiSlicesBuilder, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)

    def make_multi_slices(self, atom_histograms):
        location_phase = rfft2(atom_histograms, axes=(-2, -1))  # (n_elems, n_slices, n1, n2//2+1)
        location_phase *= self.scattering_factors[:, None, :, :]
        slices = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        np.clip(slices, a_min=1e-7, a_max=None, out=slices)
        return slices

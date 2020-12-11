from typing import Union, Tuple, List, Optional
import numpy as np
from numpy.fft import rfft2, irfft2
import logging

from .slice_builder_base import OneSliceBuilderBase, MultiSlicesBuilderBase, _symmetric_bandlimit_real

logger = logging.getLogger(__name__)


class OneSliceBuilder(OneSliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 d1: float, d2: float):
        logger.debug("using numpy OneSliceBuilder")
        super(OneSliceBuilder, self).__init__(unique_elements, n1, n2, d1, d2)

    def make_one_slice(self, atom_histograms_one_slice: np.ndarray, symmetric_bandlimit: bool = True):
        # atom_histograms_one_slice (n_elems, n1, n2)
        location_phase = rfft2(atom_histograms_one_slice, axes=(-2, -1))  # (n_elems, n1, n2//2+1)
        location_phase *= self.scattering_factors                         # (n_elems, n1, n2//2+1)
        aslice = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        if symmetric_bandlimit:
            aslice = _symmetric_bandlimit_real(aslice)
        np.clip(aslice, a_min=1e-7, a_max=None, out=aslice)
        return aslice



class MultiSlicesBuilder(MultiSlicesBuilderBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, d1: float, d2: float):
        logger.debug("using numpy MultiSlicesBuilder")
        super(MultiSlicesBuilder, self).__init__(unique_elements, n_slices, n1, n2, dz, d1, d2)

    def make_multi_slices(self, atom_histograms, symmetric_bandlimit: bool = True):
        location_phase = rfft2(atom_histograms, axes=(-2, -1))  # (n_elems, n_slices, n1, n2//2+1)
        location_phase *= self.scattering_factors[:, None, :, :]
        slices = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        if symmetric_bandlimit:
            slices = _symmetric_bandlimit_real(slices)
        np.clip(slices, a_min=1e-7, a_max=None, out=slices)
        return slices

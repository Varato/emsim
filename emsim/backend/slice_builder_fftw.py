from typing import List
import numpy as np
import logging

from .slice_builder_base import OneSliceBuilderBase, MultiSlicesBuilderBase, _symmetric_bandlimit_real

try:
    from .fftw_ext import slice_kernel
except ImportError:
    raise ImportError("cpp extension cannot be found. Compile it first")


logger = logging.getLogger(__name__)


# Here we cheat. Use numpy for one slice buiding.
class OneSliceBuilder(OneSliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 d1: float, d2: float):
        logger.debug("using fftw OneSliceBuilder")
        super(OneSliceBuilder, self).__init__(unique_elements, n1, n2, d1, d2)

    def make_one_slice(self, atom_histograms_one_slice: np.ndarray, symmetric_bandlimit: bool = True):
        # atom_histograms_one_slice (n_elems, n1, n2)
        location_phase = np.fft.rfft2(atom_histograms_one_slice, axes=(-2, -1))  # (n_elems, n1, n2//2+1)
        location_phase *= self.scattering_factors                         # (n_elems, n1, n2//2+1)
        aslice = np.fft.irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        if symmetric_bandlimit:
            aslice = _symmetric_bandlimit_real(aslice)
        np.clip(aslice, a_min=1e-13, a_max=None, out=aslice)
        return aslice


class MultiSlicesBuilder(MultiSlicesBuilderBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, d1: float, d2: float):
        logger.debug("using fftw MultiSliceBuilder")
        super(MultiSlicesBuilder, self).__init__(unique_elements, n_slices, n1, n2, dz, d1, d2)
        self.backend = slice_kernel.MultiSlicesBuilder(self.scattering_factors, n_slices, n1, n2, dz, d1, d2)

    def make_multi_slices(self, atom_histograms, symmetric_bandlimit: bool = True):
        slices = self.backend.make_multi_slices(atom_histograms.astype(np.float32))
        if symmetric_bandlimit:
            slices = _symmetric_bandlimit_real(slices)
        np.clip(slices, a_min=1e-13, a_max=None, out=slices)
        return slices

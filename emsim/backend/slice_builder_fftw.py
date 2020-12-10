from typing import List
import numpy as np
import logging

from .slice_builder_base import OneSliceBuilderBase, MultiSlicesBuilderBase

try:
    from .fftw_ext import slice_kernel
except ImportError:
    raise ImportError("cpp extension cannot be found. Compile it first")


logger = logging.getLogger(__name__)


# Here we cheat. Use numpy for one slice buiding.
class OneSliceBuilder(OneSliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 pixel_size: float):
        logger.debug("using fftw OneSliceBuilder")
        super(OneSliceBuilder, self).__init__(unique_elements, n1, n2, pixel_size)

    def make_one_slice(self, atom_histograms_one_slice):
        # atom_histograms_one_slice (n_elems, n1, n2)
        location_phase = np.fft.rfft2(atom_histograms_one_slice, axes=(-2, -1))  # (n_elems, n1, n2//2+1)
        location_phase *= self.scattering_factors               # (n_elems, n1, n2//2+1)
        aslice = np.fft.irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        np.clip(aslice, a_min=1e-7, a_max=None, out=aslice)
        return aslice


class MultiSliceBuilder(MultiSlicesBuilderBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        logger.debug("using fftw MultiSliceBuilder")
        super(MultiSliceBuilder, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)
        self.backend = dens_kernel.MultiSliceBuilder(self.scattering_factors, n_slices, n1, n2, dz, pixel_size)

    def make_multi_slices(self, atom_histograms):
        return self.backend.make_multi_slices(atom_histograms.astype(np.float32))

from typing import List
import numpy as np
import logging

from .slice_builder_base import SliceBuilderBase, SliceBuilderBatchBase

try:
    from .fftw_ext import dens_kernel
except ImportError:
    raise ImportError("cpp extension cannot be found. Compile it first")


logger = logging.getLogger(__name__)


class SliceBuilder(SliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 pixel_size: float):
        logger.debug("using fftw SliceBuilder")
        super(SliceBuilder, self).__init__(unique_elements, n1, n2, pixel_size)
        self.backend = dens_kernel.SliceBuilder(self.scattering_factors, n1, n2, pixel_size)

    def slice_gen(self, slice_atom_histograms):
        aslice = self.backend.slice_gen(slice_atom_histograms)
        return aslice


class SliceBuilderBatch(SliceBuilderBatchBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        logger.debug("using fftw SliceBuilderBatch")
        super(SliceBuilderBatch, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)
        self.backend = dens_kernel.SliceBuilderBatch(self.scattering_factors, n_slices, n1, n2, dz, pixel_size)

    def slice_gen_batch(self, atom_histograms):
        return self.backend.slice_gen_batch(atom_histograms.astype(np.float32))

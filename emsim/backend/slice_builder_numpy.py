from typing import Union, Tuple, List, Optional
import numpy as np
from numpy.fft import rfft2, irfft2
import logging

from .slice_builder_base import SliceBuilderBase, SliceBuilderBatchBase

logger = logging.getLogger(__name__)


class SliceBuilder(SliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 pixel_size: float):
        logger.debug("using numpy SliceBuilder")
        super(SliceBuilder, self).__init__(unique_elements, n1, n2, pixel_size)

    def slice_gen(self, atom_histograms):
        # atom_histograms (n_elems, n1, n2)
        location_phase = rfft2(atom_histograms, axes=(-2, -1))  # (n_elems, n1, n2//2+1)
        location_phase *= self.scattering_factors               # (n_elems, n1, n2//2+1)
        slice_ = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        np.clip(slice_, a_min=1e-7, a_max=None, out=slice_)
        return slice_



class SliceBuilderBatch(SliceBuilderBatchBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        logger.debug("using numpy SliceBuilderBatch")
        super(SliceBuilderBatch, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)

    def slice_gen_batch(self, atom_histogram):
        location_phase = rfft2(atom_histogram, axes=(-2, -1))  # (n_elems, n_slices, n1, n2//2+1)
        location_phase *= self.scattering_factors[:, None, :, :]
        slices = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        np.clip(slices, a_min=1e-7, a_max=None, out=slices)
        return slices

from typing import Union, Tuple, List, Optional
import numpy as np
from numpy.fft import rfft2, irfft2

from .slice_builder_base import SliceBuilderBatchBase


class SliceBuilderBatch(SliceBuilderBatchBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        print("using numpy SliceBuilderBatch")
        super(SliceBuilderBatch, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)

    def slice_gen_batch(self, atom_histogram):
        location_phase = rfft2(atom_histogram, axes=(-2, -1))  # (n_elems, n_slices, n1, n2//2+1)
        location_phase *= self.scattering_factors[:, None, :, :]
        slices = irfft2(np.sum(location_phase, axis=0), s=(self.n1, self.n2))
        np.clip(slices, a_min=1e-7, a_max=None, out=slices)
        return slices

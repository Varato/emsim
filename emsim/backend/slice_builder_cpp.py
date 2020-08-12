from typing import List
import numpy as np

from .slice_builder_base import SliceBuilderBatchBase

try:
    from . import dens_kernel
except ImportError:
    raise ImportError("cpp extension cannot be found. Compile it first")


class SliceBuilderBatch(SliceBuilderBatchBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        super(SliceBuilderBatch, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)
        n_elems = len(unique_elements)
        self.backend = dens_kernel.SliceBuilderBatch(self.scattering_factors, n_elems, n_slices, n1, n2, dz, pixel_size)

    def slice_gen_batch(self, atom_histograms):
        return self.backend.slice_gen_batch(atom_histograms.astype(np.float32))

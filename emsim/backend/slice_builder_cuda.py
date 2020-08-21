from typing import List
import logging

try:
    import cupy as cp
except ImportError:
    raise ImportError("the module require cupy")

try:
    from .cuda_ext import dens_kernel_cuda
except ImportError:
    raise ImportError("cuda extension cannot be found. Compile it first")

from .slice_builder_base import SliceBuilderBatchBase
from ..physics import water_num_dens


logger = logging.getLogger(__name__)

cupy_mempool = cp.get_default_memory_pool()


class SliceBuilderBatch(SliceBuilderBatchBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        logger.debug("using cuda SliceBuilderBatch")
        print("cupy mempool limit: ", cupy_mempool.get_limit())
        super(SliceBuilderBatch, self).__init__(unique_elements, n_slices, n1, n2, dz, pixel_size)
        scattering_factors = cp.asarray(self.scattering_factors, dtype=cp.float32)
        self.backend = dens_kernel_cuda.SliceBuilderBatch(scattering_factors, n_slices, n1, n2, dz, pixel_size)

    def bin_atoms(self, atom_coordinates_sorted_by_elems, unique_elements_count):
        elems_count_gpu = cp.asarray(unique_elements_count, dtype=cp.uint32)
        atom_coords_gpu = cp.asarray(atom_coordinates_sorted_by_elems, dtype=cp.float32)
        atmv_gpu = self.backend.bin_atoms(atom_coords_gpu, elems_count_gpu)
        return atmv_gpu

    def add_water(self, atom_histograms_gpu):
        vacs = cp.prod(cp.where(atom_histograms_gpu == 0, True, False), axis=0)
        # average number of water molecules in a voxel
        vox_wat_num = water_num_dens * self.pixel_size * self.pixel_size * self.dz
        box = (self.n_slice, self.n1, self.n2)

        oxygens = cp.where(vacs, cp.random.poisson(vox_wat_num, box), 0).astype(cp.int)
        hydrogens = cp.where(vacs, cp.random.poisson(vox_wat_num * 2, box), 0).astype(cp.int)

        unique_elements_list = list(self.unique_elements)
        for z, hist in [(1, hydrogens), (8, oxygens)]:
            idx = unique_elements_list.index(z)
            atom_histograms_gpu[idx] += hist
        return atom_histograms_gpu

    def slice_gen_batch(self, atom_histograms):
        print("cupy allocated: {:.2f}MB".format(cupy_mempool.total_bytes()/1024**2))
        print("cupy used total: {:.2f}MB".format(cupy_mempool.used_bytes()/1024**2))
        return self.backend.slice_gen_batch(atom_histograms)

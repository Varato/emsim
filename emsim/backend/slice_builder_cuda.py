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

from .slice_builder_base import SliceBuilderBase, SliceBuilderBatchBase
from ..physics import water_num_dens


logger = logging.getLogger(__name__)

cupy_mempool = cp.get_default_memory_pool()


class SliceBuilder(SliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 pixel_size: float):
        logger.debug("using cuda SliceBuilder")
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
        logger.debug("using cuda SliceBuilderBatch")
        logger.info(f"cupy mempool limit: {cupy_mempool.get_limit()/1024**2:.2f}MB")
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
        slices = self.backend.slice_gen_batch(atom_histograms)
        logger.info("cupy allocated: {:.2f}MB".format(cupy_mempool.total_bytes()/1024**2))
        logger.info("cupy used total: {:.2f}MB".format(cupy_mempool.used_bytes()/1024**2))
        return slices

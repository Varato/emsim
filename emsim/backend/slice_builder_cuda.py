from typing import List
import logging
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError("the module require cupy")

try:
    from .cuda_ext import slice_kernel_cuda
except ImportError:
    raise ImportError("cuda extension cannot be found. Compile it first")

from .slice_builder_base import OneSliceBuilderBase, MultiSlicesBuilderBase
from ..physics import water_num_dens


logger = logging.getLogger(__name__)

cupy_mempool = cp.get_default_memory_pool()


class OneSliceBuilder(OneSliceBuilderBase):
    def __init__(self, unique_elements: List[int], 
                 n1: int, n2: int,
                 d1: float, d2: float):
        logger.debug("using cuda OneSliceBuilder")
        super(OneSliceBuilder, self).__init__(unique_elements, n1, n2, d1, d2)
        scattering_factors = cp.asarray(self.scattering_factors, dtype=cp.float32)
        self.backend = slice_kernel_cuda.OneSliceBuilder(scattering_factors, n1, n2, d1, d2)

    def bin_atoms_one_slice(self, atom_coordinates_sorted_by_elems, unique_elements_count):
        elems_count_gpu = cp.asarray(unique_elements_count, dtype=cp.uint32)
        atom_coords_gpu = cp.asarray(atom_coordinates_sorted_by_elems, dtype=cp.float32)
        atmv_gpu = self.backend.bin_atoms_one_slice(atom_coords_gpu, elems_count_gpu)
        return atmv_gpu

    def make_one_slice(self, atom_histograms_one_slice_gpu):
        aslice_gpu = self.backend.make_one_slice(atom_histograms_one_slice_gpu)
        cp.clip(aslice_gpu, a_min=1e-7, a_max=None, out=aslice_gpu)
        return aslice_gpu


class MultiSlicesBuilder(MultiSlicesBuilderBase):
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, d1: float, d2: float):
        logger.debug("using cuda MultiSlicesBuilder")
        logger.debug(f"cupy mempool limit: {cupy_mempool.get_limit()/1024**2:.2f}MB")
        super(MultiSlicesBuilder, self).__init__(unique_elements, n_slices, n1, n2, dz, d1, d2)
        scattering_factors = cp.asarray(self.scattering_factors, dtype=cp.float32)
        self.backend = slice_kernel_cuda.MultiSlicesBuilder(scattering_factors, n_slices, n1, n2, dz, d1, d2)

    def bin_atoms_multi_slices(self, atom_coordinates_sorted_by_elems, unique_elements_count):
        elems_count_gpu = cp.asarray(unique_elements_count, dtype=cp.uint32)
        atom_coords_gpu = cp.asarray(atom_coordinates_sorted_by_elems, dtype=cp.float32)
        atmv_gpu = self.backend.bin_atoms_multi_slices(atom_coords_gpu, elems_count_gpu)
        return atmv_gpu

    def make_multi_slices(self, atom_histograms):
        slices_gpu = self.backend.make_multi_slices(atom_histograms)
        logger.debug("cupy allocated: {:.2f}MB".format(cupy_mempool.total_bytes()/1024**2))
        logger.debug("cupy used total: {:.2f}MB".format(cupy_mempool.used_bytes()/1024**2))
        cp.clip(slices_gpu, a_min=1e-7, a_max=None, out=slices_gpu)
        return slices_gpu

    def add_water(self, atom_histograms_gpu):
        vacs = cp.prod(cp.where(atom_histograms_gpu == 0, True, False), axis=0)
        # average number of water molecules in a voxel
        vox_wat_num = water_num_dens * self.d1 * self.d2 * self.dz
        box = (self.n_slice, self.n1, self.n2)

        oxygens = cp.where(vacs, cp.random.poisson(vox_wat_num, box), 0).astype(cp.int)
        hydrogens = cp.where(vacs, cp.random.poisson(vox_wat_num * 2, box), 0).astype(cp.int)

        unique_elements_list = list(self.unique_elements)
        for z, hist in [(1, hydrogens), (8, oxygens)]:
            idx = unique_elements_list.index(z)
            atom_histograms_gpu[idx] += hist
        return atom_histograms_gpu

    

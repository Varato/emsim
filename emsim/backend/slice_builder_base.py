from typing import List
import numpy as np

from .. import elem
from ..physics import water_num_dens


class SliceBuilderBatchBase:
    def __init__(self, unique_elements: List[int],
                 n_slices: int, n1: int, n2: int,
                 dz: float, pixel_size: float):
        self.n_slice = n_slices
        self.n1 = n1
        self.n2 = n2
        self.dz = dz
        self.pixel_size = pixel_size

        self.unique_elements = sorted(unique_elements)

        scattering_factors = elem.scattering_factors2d(self.unique_elements, pixel_size, (self.n1, self.n2))
        scattering_factors = np.fft.ifftshift(scattering_factors, axes=(-2, -1))
        scattering_factors = np.ascontiguousarray(scattering_factors[:, :, :self.n2//2 + 1], dtype=np.float32)
        self.scattering_factors = scattering_factors
        self.unique_elements = unique_elements

    def bin_atoms(self, atom_coordinates_sorted_by_elems, unique_elements_count):
        atmh = bin_atoms(atom_coordinates_sorted_by_elems, unique_elements_count,
                         self.n_slice, self.n1, self.n2,
                         self.dz, self.pixel_size, self.pixel_size)
        return atmh

    def add_water(self, atom_histograms):
        vacs = np.logical_and.reduce(np.where(atom_histograms == 0, True, False), axis=0)
        # average number of water molecules in a voxel
        vox_wat_num = water_num_dens * self.pixel_size*self.pixel_size*self.dz
        box = (self.n_slice, self.n1, self.n2)

        oxygens = np.where(vacs, np.random.poisson(vox_wat_num, box), 0).astype(np.float32)
        hydrogens = np.where(vacs, np.random.poisson(vox_wat_num * 2, box), 0).astype(np.float32)

        unique_elements_list = list(self.unique_elements)
        for z, hist in [(1, hydrogens), (8, oxygens)]:
            idx = unique_elements_list.index(z)
            atom_histograms[idx] += hist
        return atom_histograms

    def slice_gen_batch(self, atom_histogram):
        pass


def bin_atoms(atom_coordinates_sorted_by_elems, unique_elements_count, n0, n1, n2, d0, d1, d2):
    shape = (n0, n1, n2)
    steps = (d0, d1, d2)
    start_coord = [-steps[d] * (shape[d] // 2) for d in range(3)]
    end_coord = [start_coord[d] + shape[d] * steps[d] for d in range(3)]
    bin_edges = [np.arange(start_coord[d], end_coord[d] + steps[d], steps[d]) for d in range(3)]

    n_bins = [len(bin_edges[d]) - 1 for d in range(3)]
    n_elems = len(unique_elements_count)
    volumes = np.empty(shape=(n_elems, *n_bins), dtype=np.float32)

    m = 0
    for i in range(n_elems):
        n = unique_elements_count[i]
        if n == 0:
            volumes[i, ...] = 0
        else:
            volumes[i, ...], _ = np.histogramdd(atom_coordinates_sorted_by_elems[m:m+n], bins=bin_edges)
        m += n
    return np.ascontiguousarray(volumes, dtype=np.float32)

import unittest
import matplotlib.pyplot as plt
import numpy as np
import time

import emsim
from emsim import utils
from emsim import atoms as atm
from emsim import elem
from emsim import dens


class DensityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        # pipeline
        # mol = atm.AtomList(elements=np.array([6, 6]), coordinates=np.array([[0, -2, -2], [0, 2, 2]], dtype=np.float32))
        pdb_code = '4ear'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, data_dir)
        mol = emsim.utils.pdb.build_biological_unit(pdb_file)
        self.mol = atm.centralize(mol)
        self.voxel_size = 1.0

    def test_build_potential_fourier(self):
        pot = dens.build_potential_fourier(self.mol, self.voxel_size, box_size=70)
        print("pot  shape:", pot.shape)
        plt.imshow(pot.sum(-1))
        plt.show()

    def test_build_slices_numpy(self):
        builder = dens.get_slice_builder(backend="numpy")
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=False)
        plt.imshow(slices.sum(0))
        plt.show()

    def test_build_slices_cuda(self):
        builder = dens.get_slice_builder(backend="cuda")
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=False)
        plt.imshow(slices.sum(0).get())
        plt.show()

    def test_build_slices_cpp(self):
        builder = dens.get_slice_builder(backend="cpp")
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=False)
        plt.imshow(slices.sum(0))
        plt.show()

    def test_build_slices_fourier_cuda(self):

        slices = dens.build_slices_fourier_cuda(
            self.mol, pixel_size=self.voxel_size, thickness=4,
            lateral_size=50, add_water=True)
        plt.imshow(slices.sum(0).get())
        plt.show()

    def test_build_slices_fourier_cupy(self):
        slices = dens.build_slices_fourier_cupy(
            self.mol, pixel_size=self.voxel_size, thickness=1.2,
            lateral_size=200, n_slices=52)
        print(slices.device)
        plt.imshow(slices.sum(0).get())
        plt.show()

    def test_build_slices_fourier(self):
        t0 = time.time()
        slices_numpy = dens.build_slices_fourier(
            self.mol, pixel_size=self.voxel_size, thickness=4.,
            lateral_size=128, add_water=False)
        t1 = time.time()
        slices_cupy = dens.build_slices_fourier_cupy(
            self.mol, pixel_size=self.voxel_size, thickness=4.,
            lateral_size=128, add_water=False)
        t2 = time.time()
        slices_fftw = dens.build_slices_fourier_fftw(
            self.mol, pixel_size=self.voxel_size, thickness=4.,
            lateral_size=128, add_water=False)
        t3 = time.time()
        slices_cuda = dens.build_slices_fourier_cuda(
            self.mol, pixel_size=self.voxel_size, thickness=4.,
            lateral_size=128, add_water=False)
        t4 = time.time()

        time_numpy = t1 - t0
        time_cupy = t2 - t1
        time_fftw = t3 - t1
        time_cuda = t4 - t3
        print("slices shape:", slices_numpy.shape)

        print(f"numpy time = {time_numpy:.3f}")
        print(f"fftw time = {time_fftw:.3f}")
        print(f"cupy time = {time_cupy:.3f}")
        print(f"cuda time = {time_cuda:.3f}")

        print("difference np fftw: ", np.abs(slices_numpy - slices_fftw).max())
        print("difference np cuda: ", np.abs(slices_numpy - slices_cuda.get()).max())
        print("difference np cupy: ", np.abs(slices_numpy - slices_cupy.get()).max())

        _, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(8, 2))
        ax1.imshow(slices_numpy.sum(0))
        ax2.imshow(slices_fftw.sum(0))
        ax3.imshow(slices_cuda.sum(0).get())
        ax4.imshow(slices_cupy.sum(0).get())
        plt.show()

    def test_build_potential_fourier_water(self):
        pot = dens.build_potential_fourier(self.mol, self.voxel_size, box_size=70, add_water=True)
        print("pot  shape:", pot.shape)
        plt.imshow(pot.sum(-1))
        plt.show()

    def test_build_slices_fourier_water(self):
        slices = dens.build_slices_fourier_fftw(
            self.mol, pixel_size=self.voxel_size, thickness=1.2, lateral_size=(70, 70), add_water=True)
        print(slices.shape)
        plt.imshow(slices.sum(0))
        plt.show()


class OneAtomTestCase(unittest.TestCase):
    def setUp(self) -> None:
        elems = np.array([6])
        coords = np.array([[0, 0, 0]])
        mol = atm.AtomList(elements=elems, coordinates=coords)
        self.mol = atm.centralize(mol)
        self.voxel_size = 0.05

    def test_one_atom_potential(self):
        v_carbon = elem.potentials(elem_numbers=[6], voxel_size=self.voxel_size, radius=3.0)[0]
        dim = v_carbon.shape[0]

        v = dens.build_potential_fourier(self.mol, self.voxel_size, box_size=dim)
        plt.plot(v[dim//2 + 1:, dim//2, dim//2], label="v")
        plt.plot(v_carbon[dim//2 + 1:, dim//2, dim//2], label="v_carbon")
        plt.legend()
        plt.show()

    def test_one_atom_slice(self):
        vz_carbon = elem.projected_potentials(elem_numbers=[6], pixel_size=self.voxel_size, radius=3.0)[0]
        print(vz_carbon.shape)
        dim = vz_carbon.shape[0]

        v = dens.build_slices_fourier_fftw(self.mol, pixel_size=self.voxel_size, thickness=0.05, lateral_size=dim)
        # plt.imshow(v.sum(-1))
        plt.plot(v.sum(-1)[dim//2 + 1:, dim//2], label="vz")
        plt.plot(vz_carbon[dim//2 + 1:, dim//2], label="vz_carbon")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()

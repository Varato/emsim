import unittest
import matplotlib.pyplot as plt
import numpy as np

import emsim
from emsim import utils
from emsim import atoms as atm
from emsim import elem
from emsim import dens


class DensityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pdir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        # pipeline
        pdb_code = '1fat'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, pdir)
        mol = emsim.utils.pdb.build_biological_unit(pdb_file)
        self.mol = atm.centralize(mol)
        self.voxel_size = 2.0

    def test_build_potential_fourier(self):
        pot = dens.build_potential_fourier(self.mol, self.voxel_size, box_size=70)
        print("pot  shape:", pot.shape)
        plt.imshow(pot.sum(-1))
        plt.show()

    def test_build_slices_fourier(self):
        slices = dens.build_slices_fourier(
            self.mol, pixel_size=self.voxel_size, thickness=1.2, frame_size=(65, 70), n_slices=2)
        print(slices.shape)
        plt.imshow(slices.sum(-1))
        plt.show()

    def test_build_potential_fourier_water(self):
        pot = dens.build_potential_fourier(self.mol, self.voxel_size, box_size=70, add_water=True)
        print("pot  shape:", pot.shape)
        plt.imshow(pot.sum(-1))
        plt.show()

    def test_build_slices_fourier_water(self):
        slices = dens.build_slices_fourier(
            self.mol, pixel_size=self.voxel_size, thickness=1.2, frame_size=(70, 70), add_water=True)
        print(slices.shape)
        plt.imshow(slices.sum(-1))
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
        print(v_carbon.shape)
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

        v = dens.build_slices_fourier(self.mol, pixel_size=self.voxel_size, thickness=0.05, frame_size=dim)
        # plt.imshow(v.sum(-1))
        plt.plot(v.sum(-1)[dim//2 + 1:, dim//2], label="vz")
        plt.plot(vz_carbon[dim//2 + 1:, dim//2], label="vz_carbon")
        plt.legend()
        plt.show()



if __name__ == '__main__':
    unittest.main()

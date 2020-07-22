import unittest
import matplotlib.pyplot as plt

import emsim
from emsim import utils
from emsim import atoms as atm
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


if __name__ == '__main__':
    unittest.main()

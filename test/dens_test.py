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
        pdb_file = utils.pdb.fetch_pdb_file(pdb_code, pdir)
        mol = emsim.utils.pdb.build_biological_unit(pdb_file)
        self.mol = atm.centralize(mol)
        self.voxel_size = 2.0

    def test_fourier_builder(self):
        atmv = atm.bin_atoms(self.mol, voxel_size=self.voxel_size, box_size=60)
        pot = dens.fourier_potential_builder(atmv, projected=False)
        print("pot  shape:", pot.shape)
        print("atmv shape:", atmv.box_size)
        plt.imshow(pot.sum(0))
        plt.show()


if __name__ == '__main__':
    unittest.main()

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
        pdb_code = '4ear'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, data_dir)
        mol = emsim.utils.pdb.build_biological_unit(pdb_file)
        self.mol = atm.centralize(mol)
        self.voxel_size = 1.0

    def test_build_slices_numpy(self):
        builder = dens.get_slice_builder()
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=True)
        plt.imshow(slices.sum(0))
        plt.show()

    def test_build_slices_cuda(self):
        emsim.config.set_backend("cuda")
        builder = dens.get_slice_builder()
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=True)
        plt.imshow(slices.sum(0).get())
        plt.show()

    def test_build_slices_fftw(self):
        emsim.config.set_backend("fftw")
        builder = dens.get_slice_builder()
        slices = builder(self.mol, pixel_size=self.voxel_size, dz=1,
                         lateral_size=128, add_water=True)
        plt.imshow(slices.sum(0))
        plt.show()


if __name__ == '__main__':
    unittest.main()

import os
import unittest
import configparser
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import emsim
from emsim import utils
from emsim import atoms as atm


class AtomListTest(unittest.TestCase):
    def setUp(self):
        pdir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        pdb_code = '1fat'

        filename = utils.pdb.retrieve_pdb_file(pdb_code, pdir)
        mol = utils.pdb.build_biological_unit(filename)
        self.mol = atm.centralize(mol)
        self.voxel_size = 2.0

    def test_sort_and_count(self):
        atml, ue, uc = atm.sort_elements_and_count(self.mol, must_include_elems=[1])
        print(ue)
        print(uc)

    def test_atom_volume(self):
        atmv = atm.bin_atoms(self.mol, voxel_size=self.voxel_size)
        print(atmv.box_size)
        plt.imshow(atmv.atom_histograms[atmv.unique_elements.index(6)].sum(0))
        plt.show()

    def test_rotate_atom_volume(self):
        fig, ax = plt.subplots()

        def init():
            im = ax.imshow(np.zeros((80, 80)))
            ax.set_xlim(0, 79)
            ax.set_ylim(0, 79)
            return im

        def update(theta):
            quat = utils.rot.get_quaternion(np.array([0, 1, 1]), theta)
            mol = atm.rotate(self.mol, quat)
            atmv = atm.bin_atoms(mol, voxel_size=self.voxel_size)
            im = ax.imshow(atmv.atom_histograms[atmv.unique_elements.index(6)].sum(0))
            return im

        ani = FuncAnimation(fig, update, frames=np.linspace(0, np.pi*2, 30), init_func=init)
        plt.show()


if __name__ == '__main__':
    unittest.main()

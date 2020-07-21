import os
import unittest
import configparser
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


from emsim import utils
from emsim import atom_list as al
from emsim import rotation


class AtomListTest(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read(str(Path.home() / 'emsimConfig.ini'))
        self.data_dir = Path(config['DEFAULT']['data_dir'])
        if not self.data_dir.is_dir():
            os.mkdir(self.data_dir)

        pdb_code = '4bed'

        filename = utils.pdb.fetch_pdb_file(pdb_code, self.data_dir)
        mol = utils.pdb.build_biological_unit(filename)
        self.mol = al.centralize(mol)

    def test_bin_atoms(self):
        volume = al.bin_atoms(self.mol, voxel_size=3.0)
        print(volume.keys())
        print(volume[6].shape)

        plt.imshow(volume[6].sum(2))
        plt.show()

    def test_rotate(self):
        vox = 8.0
        fig, ax = plt.subplots()

        def init():
            im = ax.imshow(np.zeros((80, 80)))
            ax.set_xlim(0, 79)
            ax.set_ylim(0, 79)
            return im

        def update(theta):
            quat = rotation.get_quaternion(np.array([0, 1, 1]), theta)
            mol = al.rotate(self.mol, quat)
            volume = al.bin_atoms(mol, voxel_size=vox, box_size=80)
            im = ax.imshow(volume[6].sum(0))
            return im

        ani = FuncAnimation(fig, update, frames=np.linspace(0, np.pi*2, 30), init_func=init)
        plt.show()


if __name__ == '__main__':
    unittest.main()

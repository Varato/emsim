import os
import unittest
import configparser
from pathlib import Path
import matplotlib.pyplot as plt


from emsim import utils
from emsim import atom_list as al


class AtomListTest(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read(str(Path.home() / 'emsimConfig.ini'))
        self.data_dir = Path(config['DEFAULT']['data_dir'])
        if not self.data_dir.is_dir():
            os.mkdir(self.data_dir)

        pdb_code = '4bed'

        filename = utils.pdb.fetch_pdb_file(pdb_code, self.data_dir)
        self.mol = utils.pdb.build_biological_unit(filename)

    def test_bin_atoms(self):
        volume = al.bin_atoms(self.mol, voxel_size=3.0)
        print(volume.keys())
        print(volume[6].shape)

        plt.imshow(volume[6].sum(2))
        plt.show()


if __name__ == '__main__':
    unittest.main()

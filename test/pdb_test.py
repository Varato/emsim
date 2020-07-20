import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import configparser

from emsim import utils


class PdbTestCase(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read(str(Path.home()/'emsimConfig.ini'))
        self.data_dir = Path(config['DEFAULT']['data_dir'])
        if not self.data_dir.is_dir():
            os.mkdir(self.data_dir)

        self.pdb_code = ['1fat', '4bed', '1zik']
        self.filenames = []

    def test_pdb_parsing(self):
        for code in self.pdb_code:
            filename = utils.pdb.fetch_pdb_file(code, self.data_dir)
            self.filenames.append(filename)
        for filename in self.filenames:
            atom_list = utils.pdb.read_atoms(filename, sort=True)
            # print(atom_list.coordinates.shape)
            # print(atom_list.elements)
            self.assertTrue(np.all(atom_list.elements[:-1] <= atom_list.elements[1:]))


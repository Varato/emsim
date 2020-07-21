import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
import pprint

from emsim import utils

import numpy as np

np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.3f}"})


class PdbTestCase(unittest.TestCase):
    def setUp(self):
        config = configparser.ConfigParser()
        config.read(str(Path.home()/'emsimConfig.ini'))
        self.data_dir = Path(config['DEFAULT']['data_dir'])
        if not self.data_dir.is_dir():
            os.mkdir(self.data_dir)

        self.pdb_code = ['4bed']
        self.filenames = []

    def test_pdb_atoms(self):
        for code in self.pdb_code:
            filename = utils.pdb.fetch_pdb_file(code, self.data_dir, file_format='pdb')
            self.filenames.append(filename)
        for filename in self.filenames:
            atom_list = utils.pdb.read_atoms(filename, sort=True)
            print(atom_list.coordinates[:5])
            self.assertTrue(np.all(atom_list.elements[:-1] <= atom_list.elements[1:]))

    def test_cif_atoms(self):
        for code in self.pdb_code:
            filename = utils.pdb.fetch_pdb_file(code, self.data_dir, file_format='mmCif')
            self.filenames.append(filename)
        for filename in self.filenames:
            atom_list = utils.pdb.read_atoms(filename, sort=True)
            print(atom_list.coordinates[:5])
            self.assertTrue(np.all(atom_list.elements[:-1] <= atom_list.elements[1:]))

    def test_pdb_symmetry(self):
        for code in self.pdb_code:
            filename = utils.pdb.fetch_pdb_file(code, self.data_dir, file_format='pdb')
            self.filenames.append(filename)

        print('checking file', self.filenames[0])
        ops = utils.pdb.read_symmetries(self.filenames[0])
        print(ops)
        print(ops.shape)

    def test_cif_symmetry(self):
        for code in self.pdb_code:
            filename = utils.pdb.fetch_pdb_file(code, self.data_dir, file_format='mmCif')
            self.filenames.append(filename)

        print('checking file', self.filenames[0])
        ops = utils.pdb.read_symmetries(self.filenames[0])
        print(ops)
        print(ops.shape)





import os
import unittest
import numpy as np
from pathlib import Path

from emsim import utils
from emsim import io

np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.3f}"})


class PdbTestCase(unittest.TestCase):
    def setUp(self):
        self.pdir = io.data_dir.get_pdb_data_dir_from_config()
        self.pdb_code = ['4bed']
        self.filenames = []

    def test_pdb_atoms(self):
        for code in self.pdb_code:
            filename = utils.pdb.retrieve_pdb_file(code, self.pdir, file_format='pdb')
            self.filenames.append(filename)
        for filename in self.filenames:
            atom_list = utils.pdb.read_atoms(filename)
            print(atom_list.coordinates[:5])

    def test_cif_atoms(self):
        for code in self.pdb_code:
            filename = utils.pdb.retrieve_pdb_file(code, self.pdir, file_format='mmCif')
            self.filenames.append(filename)
        for filename in self.filenames:
            atom_list = utils.pdb.read_atoms(filename)
            print(atom_list.coordinates[:5])

    def test_pdb_symmetry(self):
        for code in self.pdb_code:
            filename = utils.pdb.retrieve_pdb_file(code, self.pdir, file_format='pdb')
            self.filenames.append(filename)

        print('checking file', self.filenames[0])
        ops = utils.pdb.read_symmetries(self.filenames[0])
        print(ops.shape)

    def test_cif_symmetry(self):
        for code in self.pdb_code:
            filename = utils.pdb.retrieve_pdb_file(code, self.pdir, file_format='mmCif')
            self.filenames.append(filename)

        print('checking file', self.filenames[0])
        ops = utils.pdb.read_symmetries(self.filenames[0])
        print(ops.shape)

    def test_build_biological_unit(self):
        for code in self.pdb_code:
            filename = utils.pdb.retrieve_pdb_file(code, self.pdir, file_format='mmCif')
            self.filenames.append(filename)

        al = utils.pdb.build_biological_unit(self.filenames[0])
        print(al.elements.shape)
        print(al.coordinates.shape)


if __name__ == '__main__':
    unittest.main()
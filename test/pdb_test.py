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
        config.read(Path.home()/'emsimConfig.ini')
        self.data_dir = Path(config['DEFAULT']['data_dir'])
        if not self.data_dir.is_dir():
            os.mkdir(self.data_dir)

    def test_pdb_download(self):
        utils.pdb.fetch_pdb_file('4BED', self.data_dir)

    def test_pdb_parsing(self):
        elems, coords = utils.pdb.read_atoms_and_coordinates(self.data_dir / '4BED.pdb', assemble=False)
        print(len(elems), coords.shape)

import unittest
import numpy as np

from ..emsim import atom

class AtomPotentialTestCase(unittest.TestCase):
    def test_protected_potential(self):
        elems = [1, 2, 3, 4, 5, 6]
        vs  = atom.potentials(elems, voxel_size=0.8)
        vzs = atom.projected_potentials(elems, voxel_size=0.8)
        self.assertTrue(np.all(vs.sum() == vzs))

import unittest
import numpy as np
import matplotlib.pyplot as plt

from emsim import atom

class AtomPotentialTestCase(unittest.TestCase):
    def test_protected_potential(self):
        """
        test for the fact:
            projected_potential ~ potential.sum(-1) * voxel_size when voxel_size is small
        """
        elems = [1, 2, 3, 4, 5, 6]
        voxel_size = 0.1
        vs  = atom.potentials(elems, voxel_size=voxel_size)
        vzs = atom.projected_potentials(elems, voxel_size=voxel_size)
        vzs_ = vs.sum(1) * voxel_size

        dim = vzs.shape[1]

        k = 5
        # avoid atom center when comparing
        a1 = vzs[k, dim//2,  dim//2+1:]
        a2 = vzs_[k, dim//2, dim//2+1:]

        # ax = plt.gca()
        # ax.plot(a1,  label='projected')
        # ax.plot(a2, label='summed 3D')
        # ax.plot(np.abs(a1 - a2)/a1, label='diff')
        # ax.legend()

        # plt.show()

        self.assertTrue(np.all(np.abs(a1-a2)/a1 < 0.1 * voxel_size))


if __name__ == "__main__":
    unittest.main()
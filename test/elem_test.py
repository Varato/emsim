import unittest
import numpy as np
import matplotlib.pyplot as plt

from emsim import elem


class AtomPotentialTestCase(unittest.TestCase):
    def setUp(self):
        self.elems = [1, 2, 3, 4, 5, 6]
        self.voxel_size = 1.0

    def test_atom_number(self):
        res = [elem.number(s) for s in ['H', 'He', 'Li', 'Be', 'B']]
        self.assertEqual(res, [1, 2, 3, 4, 5])

    def test_protected_potential(self):
        """
        test for the fact:
            projected_potential ~ potential.sum(-1) * voxel_size when voxel_size is small
        """
        vs = elem.potentials(self.elems, voxel_size=self.voxel_size)
        vzs = elem.projected_potentials(self.elems, voxel_size=self.voxel_size)
        vzs_ = vs.sum(1) * self.voxel_size

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

        self.assertTrue(np.all(np.abs(a1-a2)/a1 < 0.1 * self.voxel_size))

    def test_projected_scattering_factor(self):
        fq  = elem.scattering_factors(self.elems, self.voxel_size, size=21)
        fqz = elem.scattering_factors2d(self.elems, self.voxel_size, size=21)
        # print(fq.shape)
        # print(fqz.shape)

        # _, ax = plt.subplots(ncols=1)
        # ax.plot(fqz[5, :, 10])
        # ax.plot(fq[5, 10, :, 10])
        # plt.show()


if __name__ == "__main__":
    unittest.main()

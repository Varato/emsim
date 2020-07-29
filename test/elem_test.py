import unittest
import numpy as np
import matplotlib.pyplot as plt

from emsim import elem


class AtomPotentialTestCase(unittest.TestCase):
    def setUp(self):
        self.elems = [6, 14, 29, 79, 92]
        self.labels = ['C', 'Si', 'Cu', 'Au', 'U']

        self.voxel_size = 1.0

    def test_atom_number(self):
        res = [elem.number(s) for s in ['H', 'He', 'Li', 'Be', 'B']]
        self.assertEqual(res, [1, 2, 3, 4, 5])

    def test_potential_function(self):
        r = np.linspace(0.001, 0.5, 500)
        pot = elem.potential_function(self.elems)
        vs = pot(r)
        for i in range(len(self.elems)):
            plt.plot(r, vs[i], label=self.labels[i])
        plt.xlim([0, 0.5])
        plt.ylim([0, 20])
        plt.xticks(np.arange(0, 0.55, 0.05))
        plt.yticks(np.arange(0, 22, 2))
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_projected_potential_function(self):
        r = np.linspace(0.001, 0.5, 500)
        pot = elem.projected_potential_function(self.elems)
        vs = pot(r) * 1000
        for i in range(len(self.elems)):
            plt.plot(r, vs[i], label=self.labels[i])
        plt.xlim([0, 0.5])
        plt.ylim([0, 20])
        plt.xticks(np.arange(0, 0.55, 0.05))
        plt.yticks(np.arange(0, 5500, 500))
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_scattering_factor_function(self):
        q = np.linspace(0.001, 2, 500)
        feq = elem.scattering_factor_function(self.elems)
        fs = feq(q)
        for i in range(len(self.elems)):
            plt.plot(q, fs[i], label=self.labels[i])
        plt.xlim([0, 2])
        plt.ylim([0, 20])
        plt.xticks(np.arange(0, 2.2, 0.2))
        plt.yticks(np.arange(0, 22, 2))
        plt.grid(True)
        plt.legend()
        plt.show()

    def test_projected_potential(self):
        vzs = elem.projected_potentials(self.elems, pixel_size=self.voxel_size)
        _, axes = plt.subplots(ncols=len(self.elems))
        for i in range(len(self.elems)):
            axes[i].imshow(vzs[i])
        plt.show()

    def test_protected_potential(self):
        """
        test for the fact:
            projected_potential ~ potential.sum(-1) * voxel_size when voxel_size is small
        """
        vs = elem.potentials(self.elems, voxel_size=self.voxel_size)
        vzs = elem.projected_potentials(self.elems, pixel_size=self.voxel_size)
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

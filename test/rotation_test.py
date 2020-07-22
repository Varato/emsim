import unittest

from emsim.utils import rot as rot


class RotationTestCase(unittest.TestCase):
    def test_quaternion_gen(self):
        quats = rot.random_uniform_quaternions(10)
        self.assertEqual(quats.shape, (10, 4))

    def test_get_rotation_matrix(self):
        quats = rot.random_uniform_quaternions(1)
        mats = rot.get_rotation_mattrices(quats)
        self.assertEqual(mats.shape, (1, 3, 3))


if __name__ == '__main__':
    unittest.main()

import unittest
import math

from emsim import physics


class PhysicsTestCase(unittest.TestCase):
    def test_interaction_parameter(self):
        beam_energy_kev = 200
        wave_length = physics.electron_wave_length_angstrom(beam_energy_kev)
        gamma = physics.electron_relativity_gamma(beam_energy_kev)
        sigma = physics.interaction_parameter(beam_energy_kev)

        c1 = wave_length * gamma
        c2 = sigma * 2 * math.pi * physics.a0 * physics.e
        self.assertTrue(abs(c1-c2*1e8)<1e-5)


if __name__ == "__main__":
    unittest.main()
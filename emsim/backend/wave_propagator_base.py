from typing import Tuple
from abc import ABCMeta
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


from ..physics import electron_wave_length_angstrom, electron_relativity_gamma


class WavePropagatorBase:
    def __init__(self, n1: int, n2: int, d1: float, d2: float, beam_energy_kev: float):
        self.wave_shape = (n1, n2)
        self.d1, self.d2 = d1, d2

        self.wave_length = electron_wave_length_angstrom(beam_energy_kev)
        self.relativity_gamma = electron_relativity_gamma(beam_energy_kev)

        qx, qy = WavePropagatorBase._make_mesh_grid_fourier_space(n1, n2, d1, d2)
        self.q_mgrid = np.sqrt(qx * qx + qy * qy)

        # 1/3 filtering
        q_max = 0.5 / max(d1, d2)
        self.fil = ifftshift(np.where(self.q_mgrid <= q_max * 0.6667, 1., 0.))

    def init_wave(self, electron_dose: float):
        n_e = electron_dose * self.d1 * self.d2
        wave_in = np.ones(self.wave_shape, dtype=np.complex64)
        wave_in *= np.sqrt(n_e) / np.abs(wave_in)
        return wave_in

    def slice_transmit(self, wave, aslice):
        pass

    def space_propagate(self, wave, dz):
        pass

    def multislice_propagate(self, wave: np.ndarray, slices: np.ndarray, dz: float):
        pass

    def singleslice_propagate(self, wave: np.ndarray, aslice: np.ndarray, dz: float):
        pass

    def lens_propagate(self, wave: np.ndarray, cs_mm: float, defocus: float, aperture: float):
        pass

    @staticmethod
    def _make_mesh_grid_fourier_space(n1: int, n2: int, d1: float, d2: float):
        qx_max = 0.5/d1
        qy_max = 0.5/d2
        qx_range = np.linspace(-qx_max, qx_max, n1)
        qy_range = np.linspace(-qy_max, qy_max, n2)
        qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
        return qx, qy

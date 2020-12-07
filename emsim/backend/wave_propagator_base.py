from typing import Tuple
from abc import ABCMeta
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


from ..physics import electron_wave_length_angstrom, electron_relativity_gamma


class WavePropagatorBase:
    def __init__(self, shape: Tuple[int, int], pixel_size: float, beam_energy_kev: float):
        self.wave_shape = shape
        self.pixel_size = pixel_size

        self.wave_length = electron_wave_length_angstrom(beam_energy_kev)
        self.relativity_gamma = electron_relativity_gamma(beam_energy_kev)

        qx, qy = WavePropagatorBase._make_mesh_grid_fourier_space(pixel_size, self.wave_shape)
        self.q_mgrid = np.sqrt(qx * qx + qy * qy)

        # 1/3 filtering
        q_max = 0.5 / self.pixel_size
        self.fil = ifftshift(np.where(self.q_mgrid <= q_max * 0.6667, 1., 0.))

    def init_wave(self, electron_dose: float):
        n_e = electron_dose * self.pixel_size ** 2
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
    def _make_mesh_grid_fourier_space(pixel_size: float, size: Tuple[int, int]):
        q_max = 0.5/pixel_size
        qx_range = np.linspace(-q_max, q_max, size[0])
        qy_range = np.linspace(-q_max, q_max, size[1])
        qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
        return qx, qy

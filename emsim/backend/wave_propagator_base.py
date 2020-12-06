from typing import Tuple
from abc import ABCMeta
import numpy as np

from ..physics import electron_wave_length_angstrom, electron_relativity_gamma


class WavePropagatorBase:
    def __init__(self, shape: Tuple[int, int], pixel_size: float, beam_energy_kev: float):
        self.wave_shape = shape
        self.pixel_size = pixel_size

        self.wave_length = electron_wave_length_angstrom(beam_energy_kev)
        self.relativity_gamma = electron_relativity_gamma(beam_energy_kev)

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

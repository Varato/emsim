try:
    from .fftw_ext import em_kernel
except ImportError:
    raise ImportError("cannot import cpp extension em_kernel.*.pyd")

from typing import Tuple
import numpy as np

from .wave_propagator_base import WavePropagatorBase


class WavePropagator(WavePropagatorBase):
    def __init__(self, shape: Tuple[int, int], pixel_size: float, beam_energy_key: float):
        print("using fftw WavePropagator")
        super(WavePropagator, self).__init__(shape, pixel_size, beam_energy_key)
        self.backend = em_kernel.WavePropagator(shape[0], shape[1],
                                                pixel_size, self.wave_length, self.relativity_gamma)

    def singleslice_propagate(self, wave_in: np.ndarray, aslice, dz: float):
        return self.backend.singleslice_propagate(wave_in, aslice, dz)

    def multislice_propagate(self, wave_in: np.ndarray, slices: np.ndarray, dz: float):
        return self.backend.multislice_propagate(wave_in, slices, dz)

    def lens_propagate(self, wave_in, cs_mm, defocus, aperture):
        return self.backend.lens_propagate(wave_in, cs_mm, defocus, aperture)

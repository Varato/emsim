try:
    from .fftw_ext import wave_kernel
except ImportError:
    raise ImportError("cannot import cpp extension em_kernel.*.pyd")

from typing import Tuple
import numpy as np
import logging

from .wave_propagator_base import WavePropagatorBase


logger = logging.getLogger(__name__)


class WavePropagator(WavePropagatorBase):
    def __init__(self, n1: int, n2: int, d1: float, d2: float, beam_energy_key: float):
        logger.debug("using fftw WavePropagator")
        super(WavePropagator, self).__init__(n1, n2, d1, d2, beam_energy_key)
        self.backend = wave_kernel.WavePropagator(n1, n2, d1, d2,
                                                  self.wave_length, self.relativity_gamma)

    def slice_transmit(self, wave: np.ndarray, aslice: np.ndarray):
        return self.backend.slice_transmit(wave, aslice)

    def space_propagate(self, wave: np.ndarray, dz: float):
        return self.backend.space_propagate(wave, dz)

    def singleslice_propagate(self, wave: np.ndarray, aslice: np.ndarray, dz: float):
        return self.backend.singleslice_propagate(wave, aslice, dz)

    def multislice_propagate(self, wave: np.ndarray, slices: np.ndarray, dz: float):
        return self.backend.multislice_propagate(wave, slices, dz)

    def lens_propagate(self, wave, cs_mm, defocus, aperture):
        return self.backend.lens_propagate(wave, cs_mm, defocus, aperture)

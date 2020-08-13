try:
    from .cuda_ext import em_kernel_cuda
except ImportError:
    raise ImportError("cannot import cpp extension em_kernel.*.pyd")

from typing import Tuple
import cupy as cp

from .wave_propagator_base import WavePropagatorBase


class WavePropagator(WavePropagatorBase):
    def __init__(self, wave_shape: Tuple[int, int], pixel_size: float, beam_energy_key: float):
        print("using cuda WavePropagator")
        super(WavePropagator, self).__init__(wave_shape, pixel_size, beam_energy_key)
        self.backend = em_kernel_cuda.WavePropagator(wave_shape[0], wave_shape[1],
                                                     pixel_size, self.wave_length, self.relativity_gamma)

    def init_wave(self, electron_dose: float):
        n_e = electron_dose * self.pixel_size ** 2
        wave = cp.ones(self.wave_shape, dtype=cp.complex64)
        wave *= cp.sqrt(n_e) / cp.abs(wave)
        return wave

    def singleslice_propagate(self, wave_in: cp.ndarray, aslice: cp.ndarray, dz: float):
        return self.backend.singleslice_propagate(wave_in, aslice, dz)

    def multislice_propagate(self, wave_in: cp.ndarray, slices: cp.ndarray, dz: float):
        return self.backend.multislice_propagate(wave_in, slices, dz)

    def lens_propagate(self, wave_in: cp.ndarray, cs_mm: float, defocus: float, aperture: float):
        return self.backend.lens_propagate(wave_in, cs_mm, defocus, aperture)

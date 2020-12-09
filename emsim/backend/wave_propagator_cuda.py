try:
    from .cuda_ext import wave_kernel_cuda
except ImportError:
    raise ImportError("cannot import cpp extension em_kernel.*.pyd")

from typing import Tuple
import cupy as cp
import numpy as np
import logging

from .wave_propagator_base import WavePropagatorBase


logger = logging.getLogger(__name__)

def assure_cupy_array(arr):
    xp = cp.get_array_module(arr)
    if xp is np:
        return cp.asarray(arr)
    return arr

class WavePropagator(WavePropagatorBase):
    def __init__(self, shape: Tuple[int, int], pixel_size: float, beam_energy_key: float):
        logger.debug("using cuda WavePropagator")
        super(WavePropagator, self).__init__(shape, pixel_size, beam_energy_key)
        self.backend = wave_kernel_cuda.WavePropagator(shape[0], shape[1],
                                                     pixel_size, self.wave_length, self.relativity_gamma)

    def init_wave(self, electron_dose: float):
        n_e = electron_dose * self.pixel_size ** 2
        wave = cp.ones(self.wave_shape, dtype=cp.complex64)
        wave *= cp.sqrt(n_e) / cp.abs(wave)
        return wave

    def slice_transmit(self, wave: cp.ndarray, aslice: cp.ndarray):
        wave = assure_cupy_array(wave)
        aslice = assure_cupy_array(aslice)
        return self.backend.slice_transmit(wave, aslice)

    def space_propagate(self, wave: cp.ndarray, dz: float):
        wave = assure_cupy_array(wave)
        return self.backend.space_propagate(wave, dz)

    def singleslice_propagate(self, wave: cp.ndarray, aslice: cp.ndarray, dz: float):
        wave = assure_cupy_array(wave)
        aslice = assure_cupy_array(aslice)
        return self.backend.singleslice_propagate(wave, aslice, dz)

    def multislice_propagate(self, wave: cp.ndarray, slices: cp.ndarray, dz: float):
        wave = assure_cupy_array(wave)
        aslice = assure_cupy_array(slices)
        return self.backend.multislice_propagate(wave, slices, dz)

    def lens_propagate(self, wave: cp.ndarray, cs_mm: float, defocus: float, aperture: float):
        wave = assure_cupy_array(wave)
        return self.backend.lens_propagate(wave, cs_mm, defocus, aperture)

from typing import Tuple
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
import logging

from ..physics import mtf
from .wave_propagator_base import WavePropagatorBase

logger = logging.getLogger(__name__)


class WavePropagator(WavePropagatorBase):
    def __init__(self, shape: Tuple[int, int], pixel_size: float, beam_energy_key: float):
        logger.debug("using numpy WavePropagator")
        super(WavePropagator, self).__init__(shape, pixel_size, beam_energy_key)
        

    def slice_transmit(self, wave: np.ndarray, aslice: np.ndarray):
        t = np.exp(1j * self.relativity_gamma * self.wave_length * aslice)
        # t = ifft2(fft2(transmission_functions) * self.fil)
        return wave * t
    
    def space_propagate(self, wave: np.ndarray, dz: float):
        spatial_propagator = np.exp(-1j * self.wave_length * np.pi * dz * self.q_mgrid ** 2)
        return ifft2(ifftshift(spatial_propagator) * fft2(wave))

    def multislice_propagate(self, wave_in, slices, dz: float):
        n_slices = slices.shape[0]

        spatial_propagator = np.exp(-1j * self.wave_length * np.pi * dz * self.q_mgrid ** 2)
        transmission_functions = np.exp(1j * self.relativity_gamma * self.wave_length * slices)

        psi = wave_in
        for s in range(n_slices):
            psi = ifft2(ifftshift(spatial_propagator) * fft2(psi * transmission_functions[s]) * self.fil)
        return psi

    def singleslice_propagate(self, wave_in, aslice, dz: float):
        spatial_propagator = np.exp(-1j * self.wave_length * np.pi * dz * self.q_mgrid ** 2)
        transmission_functions = np.exp(1j * self.relativity_gamma * self.wave_length * aslice)
        return ifft2(ifftshift(spatial_propagator) * fft2(wave_in * transmission_functions) * self.fil)

    def lens_propagate(self, wave_in, cs_mm, defocus, aperture):
        h = mtf(self.wave_length, cs_mm, defocus)(self.q_mgrid)
        aper = np.where(self.q_mgrid < aperture / self.wave_length, 1., 0.)
        h *= aper
        return ifft2(ifftshift(h) * fft2(wave_in))

    

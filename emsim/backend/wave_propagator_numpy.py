from typing import Tuple
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from ..physics import mtf
from .wave_propagator_base import WavePropagatorBase


class WavePropagator(WavePropagatorBase):
    def __init__(self, wave_shape: Tuple[int, int], pixel_size: float, beam_energy_key: float):
        print("using numpy WavePropagator")
        super(WavePropagator, self).__init__(wave_shape, pixel_size, beam_energy_key)
        qx, qy = WavePropagator._make_mesh_grid_fourier_space(pixel_size, self.wave_shape)
        self.q_mgrid = np.sqrt(qx * qx + qy * qy)

        # 1/3 filtering
        q_max = 0.5 / self.pixel_size
        self.fil = ifftshift(np.where(self.q_mgrid <= q_max * 0.6667, 1., 0.))

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
        return ifft2(ifftshift(h) * fft2(wave_in) * ifftshift(aper))

    @staticmethod
    def _make_mesh_grid_fourier_space(pixel_size: float, size: Tuple[int, int]):
        q_max = 0.5/pixel_size
        qx_range = np.linspace(-q_max, q_max, size[0])
        qy_range = np.linspace(-q_max, q_max, size[1])
        qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
        return qx, qy

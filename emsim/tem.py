from typing import Tuple, Union
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2

from .physics import electron_wave_length_angstrom, mtf, aberration, electron_relativity_gamma
from . import wave


class Specimen(object):
    def __init__(self, slices, pixel_size, dz):
        self.slices = slices
        self.pixel_size = pixel_size
        self.dz = dz

    @property
    def lateral_size(self):
        return self.slices.shape[1:]

    @property
    def num_slices(self):
        return self.slices.shape[0]

    def __iter__(self):
        self._s = 0
        return self

    def __next__(self):
        if self._s < self.num_slices:
            slc = self.slices[self._s, ...]
            self._s += 1
            return slc
        else:
            raise StopIteration


class TEM(object):
    def __init__(self, electron_dose: float, beam_energy_kev: float, cs_mm: float, defocus: float, aperture: float):
        self.electron_dose = electron_dose
        self.beam_energy_kev = beam_energy_kev
        self.cs_mm = cs_mm
        self.defocus = defocus
        self.aperture = aperture

    @property
    def relativity_gamma(self):
        return electron_relativity_gamma(self.beam_energy_kev)

    @property
    def wave_length(self):
        return electron_wave_length_angstrom(self.beam_energy_kev)

    def get_mtf_function(self):
        return mtf(self.wave_length, self.cs_mm, self.defocus)

    def get_ctf_function(self):
        return lambda k: -mtf(self.wave_length, self.cs_mm, self.defocus)(k).imag

    def get_aberration_function(self):
        return aberration(self.wave_length, self.cs_mm, self.defocus)

    def get_wave_propagator(self, wave_shape: Union[int, Tuple[int, int]], pixel_size: Union[float, Tuple[float, float]]):
        """
        Get a wave propagator instance by giving wave_shape and pixel_size.

        Parameters
        ----------
        wave_shape: Union[int, Tuple[int, int]]
        pixel_size: Union[float, Tuple[float, float]]

        Returns
        -------
        emsim.wave.WavePropagator instance.
        """
        if type(wave_shape) is int:
            wave_shape = (wave_shape, )*2
        if type(pixel_size) is float:
            pixel_size = (pixel_size, )*2
        return wave.WavePropagator(wave_shape, pixel_size, self.beam_energy_kev, 
            self.electron_dose, self.cs_mm, self.defocus, self.aperture)

    def __repr__(self):
        return f"{{TEM | beam_energy = {self.beam_energy_kev:.2f}keV, " \
               f"cs = {self.cs_mm:.2f}mm, " \
               f"defocus = {0.1*self.defocus:.2f}nm, " \
               f"aperture = {self.aperture/np.pi * 180:.2f} deg}}"

    def __str__(self):
        return self.__repr__()


def band_limit_specimen(specimen: Specimen) -> Specimen:
    s1, s2 = specimen.lateral_size
    r = min([s1, s2]) // 2
    x, y = np.mgrid[0:s1, 0:s2]
    fil = ifftshift(np.where((x - s1 // 2) ** 2 + (y - s2 // 2) ** 2 <= 4 * r * r / 9, 1., 0.))
    filtered_slices = ifft2(fft2(specimen.slices, axes=(0, 1)) * fil[..., None], axes=(0, 1)).real
    return Specimen(filtered_slices, specimen.pixel_size, specimen.dz)
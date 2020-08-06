from typing import Generator, Tuple, Optional, Union
import numpy as np

from . import dens
from . import em
from . import wave
from . import atoms as atm


class MultislicePipe(object):
    def __init__(self,
                 microscope: em.EM,
                 mol: atm.AtomList,
                 resolution: float,
                 slice_thickness: float,
                 add_water: bool = True,
                 roi: Optional[Union[int, Tuple[int, int]]] = None,
                 n_slices: Optional[int] = None):
        self._resolution = resolution
        self._pixel_size = 0.5 * resolution
        self.microscope = microscope
        self.mol = mol
        self.slice_thickness = slice_thickness
        self.add_water = add_water

        self.roi = roi
        self.n_slices = n_slices

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        self._pixel_size = 0.5 * res
        self._resolution = res

    def run(self, back_end='cuda'):
        if back_end == 'cuda':
            return self._image_cuda()
        elif back_end == 'numpy':
            return self._image_np()
        elif back_end == "fftw":
            return self._image_fftw()
        else:
            raise ValueError(f"unrecognized backend {back_end}. back_end must be one of `cuda`, `numpy` or `fftw`")

    def _exit_wave_np(self):
        slices = dens.build_slices_fourier(self.mol,
                                           self._pixel_size,
                                           self.slice_thickness,
                                           self.roi,
                                           self.n_slices,
                                           self.add_water)
        init_wave = wave.init_wave(self.microscope.electron_dose, self._pixel_size, self.roi)
        exit_wave = wave.multislice_propagate(init_wave, slices,
                                              self._pixel_size, self.slice_thickness,
                                              self.microscope.wave_length, self.microscope.relativity_gamma)
        return exit_wave

    def _image_wave_np(self):
        exit_wave = self._exit_wave_np()
        image_wave = wave.lens_propagate(exit_wave, self._pixel_size,
                                         self.microscope.wave_length, self.microscope.cs_mm,
                                         self.microscope.defocus, self.microscope.aperture)
        return image_wave

    def _image_np(self):
        image_wave = self._image_wave_np()
        return image_wave.real ** 2 + image_wave.imag ** 2

    def _exit_wave_fftw(self):
        slices = dens.build_slices_fourier_fftw(self.mol,
                                                self._pixel_size,
                                                self.slice_thickness,
                                                self.roi,
                                                self.n_slices,
                                                self.add_water)

        init_wave = wave.init_wave(self.microscope.electron_dose, self._pixel_size, self.roi)

        exit_wave = wave.multislice_propagate_fftw(init_wave, slices,
                                                   self._pixel_size, self.slice_thickness,
                                                   self.microscope.wave_length, self.microscope.relativity_gamma)
        return exit_wave

    def _image_wave_fftw(self):
        exit_wave = self._exit_wave_fftw()
        image_wave = wave.lens_propagate_fftw(exit_wave, self._pixel_size,
                                              self.microscope.wave_length,
                                              self.microscope.cs_mm,
                                              self.microscope.defocus,
                                              self.microscope.aperture)
        return image_wave

    def _image_fftw(self):
        image_wave = self._image_wave_fftw()
        return image_wave.real ** 2 + image_wave.imag ** 2

    def _exit_wave_cuda(self):
        slices = dens.build_slices_fourier_cuda(self.mol,
                                                self._pixel_size,
                                                self.slice_thickness,
                                                self.roi,
                                                self.n_slices,
                                                self.add_water)

        init_wave = wave.init_wave_cuda(self.microscope.electron_dose, self._pixel_size, self.roi)

        exit_wave = wave.multislice_propagate_cuda(init_wave, slices,
                                                   self._pixel_size, self.slice_thickness,
                                                   self.microscope.wave_length, self.microscope.relativity_gamma)
        return exit_wave

    def _image_wave_cuda(self):
        exit_wave = self._exit_wave_cuda()
        image_wave = wave.lens_propagate_cuda(exit_wave, self._pixel_size,
                                              self.microscope.wave_length,
                                              self.microscope.cs_mm,
                                              self.microscope.defocus,
                                              self.microscope.aperture)
        return image_wave

    def _image_cuda(self):
        image_wave = self._image_wave_cuda()
        return image_wave.real ** 2 + image_wave.imag ** 2




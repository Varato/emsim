from typing import Tuple, Optional, Union

from . import dens
from . import em
from . import wave
from . import atoms as atm


class PipeBase(object):
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

        if type(roi) is int:
            self.roi = (roi, roi)
        else:
            self.roi = roi
        self.n_slices = n_slices

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        self._pixel_size = 0.5 * res
        self._resolution = res

    def exit_wave(self):
        pass

    def image_wave(self):
        pass

    def image(self):
        pass


class PipeNumpy(PipeBase):
    def __init__(self, *args, **kwargs):
        super(PipeNumpy, self).__init__(*args, **kwargs)
        self.wave_propagator = wave.WavePropagator(self.roi, self._pixel_size,
                                                   self.microscope.wave_length,
                                                   self.microscope.relativity_gamma,
                                                   backend="numpy")

    def exit_wave(self):
        slices = dens.build_slices_fourier(self.mol,
                                           self._pixel_size,
                                           self.slice_thickness,
                                           self.roi,
                                           self.n_slices,
                                           self.add_water)
        init_wave = wave.init_wave(self.microscope.electron_dose, self._pixel_size, self.roi)
        exit_wave = self.wave_propagator.multislice_propagate(init_wave, slices, self.slice_thickness)
        return exit_wave

    def image_wave(self):
        exit_wave = self.exit_wave()
        image_wave = self.wave_propagator.lens_propagate(exit_wave,
                                                         self.microscope.cs_mm,
                                                         self.microscope.defocus,
                                                         self.microscope.aperture)
        return image_wave

    def image(self):
        image_wave = self.image_wave()
        return image_wave.real ** 2 + image_wave.imag ** 2


class PipeCuda(PipeBase):
    def __init__(self, *args, **kwargs):
        super(PipeCuda, self).__init__(*args, **kwargs)
        self.wave_propagator = wave.WavePropagator(self.roi, self._pixel_size,
                                                   self.microscope.wave_length,
                                                   self.microscope.relativity_gamma)

    def exit_wave(self):
        slices = dens.build_slices_fourier_cuda(self.mol,
                                                self._pixel_size,
                                                self.slice_thickness,
                                                self.roi,
                                                self.n_slices,
                                                self.add_water)

        init_wave = wave.init_wave_cuda(self.microscope.electron_dose, self._pixel_size, self.roi)
        exit_wave = self.wave_propagator.multislice_propagate(init_wave, slices, self.slice_thickness)
        return exit_wave

    def image_wave(self):
        exit_wave = self.exit_wave()
        image_wave = self.wave_propagator.lens_propagate(exit_wave,
                                                         self.microscope.cs_mm,
                                                         self.microscope.defocus,
                                                         self.microscope.aperture)
        return image_wave

    def image(self):
        image_wave = self.image_wave()
        return image_wave.real ** 2 + image_wave.imag ** 2

    # def _exit_wave_fftw(self):
    #     slices = dens.build_slices_fourier_fftw(self.mol,
    #                                             self._pixel_size,
    #                                             self.slice_thickness,
    #                                             self.roi,
    #                                             self.n_slices,
    #                                             self.add_water)
    #
    #     init_wave = wave.init_wave(self.microscope.electron_dose, self._pixel_size, self.roi)
    #
    #     exit_wave = wave.multislice_propagate_fftw(init_wave, slices,
    #                                                self._pixel_size, self.slice_thickness,
    #                                                self.microscope.wave_length, self.microscope.relativity_gamma)
    #     return exit_wave
    #
    # def _image_wave_fftw(self):
    #     exit_wave = self._exit_wave_fftw()
    #     image_wave = wave.lens_propagate_fftw(exit_wave, self._pixel_size,
    #                                           self.microscope.wave_length,
    #                                           self.microscope.cs_mm,
    #                                           self.microscope.defocus,
    #                                           self.microscope.aperture)
    #     return image_wave
    #
    # def _image_fftw(self):
    #     image_wave = self._image_wave_fftw()
    #     return image_wave.real ** 2 + image_wave.imag ** 2
    #
    # def _exit_wave_cuda(self):
    #     slices = dens.build_slices_fourier_cuda(self.mol,
    #                                             self._pixel_size,
    #                                             self.slice_thickness,
    #                                             self.roi,
    #                                             self.n_slices,
    #                                             self.add_water)
    #
    #     init_wave = wave.init_wave_cuda(self.microscope.electron_dose, self._pixel_size, self.roi)
    #
    #     exit_wave = self.wave_propagator_cuda.multislice_propagate(init_wave, slices, self.slice_thickness)
    #     return exit_wave
    #
    # def _image_wave_cuda(self):
    #     exit_wave = self._exit_wave_cuda()
    #     image_wave = wave.lens_propagate_cuda(exit_wave, self._pixel_size,
    #                                           self.microscope.wave_length,
    #                                           self.microscope.cs_mm,
    #                                           self.microscope.defocus,
    #                                           self.microscope.aperture)
    #     return image_wave
    #
    # def _image_cuda(self):
    #     image_wave = self._image_wave_cuda()
    #     return image_wave.real ** 2 + image_wave.imag ** 2




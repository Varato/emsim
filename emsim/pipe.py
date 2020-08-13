from typing import Tuple, Optional, Union

from . import dens
from . import em
from . import wave
from . import atoms as atm
from . import config


class Pipe(object):
    def __init__(self,
                 microscope: em.EM,
                 mol: atm.AtomList,
                 resolution: float,
                 slice_thickness: float,
                 add_water: bool = True,
                 roi: Optional[Union[int, Tuple[int, int]]] = None,
                 n_slices: Optional[int] = None,
                 backend="numpy"):
        self._resolution = resolution
        self._pixel_size = 0.5 * resolution
        self.microscope = microscope
        self.mol = atm.centralize(mol)  # put the molecule in the center of roi
        self.slice_thickness = slice_thickness
        self.add_water = add_water

        if type(roi) is int:
            self.roi = (roi, roi)
        else:
            self.roi = roi
        self.n_slices = n_slices

        self.slice_builder = dens.get_slice_builder()
        self.wave_propagator = wave.get_wave_propagator()(self.roi, self._pixel_size, self.microscope.beam_energy_kev)

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        self._pixel_size = 0.5 * res
        self._resolution = res

    def set_backend(self, backend="numpy"):
        config.set_backend(backend)
        self.slice_builder = dens.get_slice_builder()
        self.wave_propagator = wave.get_wave_propagator()(self.roi, self._pixel_size, self.microscope.beam_energy_kev)

    def run(self):
        slices = self.build_slices()
        init_wave = self.init_wave()
        exit_wave = self.exit_wave(init_wave, slices)
        image_wave = self.image_wave(exit_wave)
        return self.image(image_wave)

    def build_slices(self):
        slices = self.slice_builder(self.mol,
                                    pixel_size=self._pixel_size,
                                    dz=self.slice_thickness,
                                    lateral_size=self.roi,
                                    add_water=self.add_water)
        return slices

    def init_wave(self):
        return self.wave_propagator.init_wave(self.microscope.electron_dose)

    def exit_wave(self, wave_in, slices):
        return self.wave_propagator.multislice_propagate(wave_in, slices, self.slice_thickness)

    def image_wave(self, exit_wave):
        return self.wave_propagator.lens_propagate(exit_wave,
                                                   self.microscope.cs_mm,
                                                   self.microscope.defocus,
                                                   self.microscope.aperture)

    @staticmethod
    def image(image_wave):
        return image_wave.real ** 2 + image_wave.imag ** 2

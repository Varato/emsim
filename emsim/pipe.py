from typing import Tuple, Optional, Union

from . import dens
from . import em
from . import wave
from . import atoms as atm


class Pipe(object):
    def __init__(self,
                 microscope: em.EM,
                 resolution: float,
                 slice_thickness: float,
                 roi: Optional[Union[int, Tuple[int, int]]] = None,
                 n_slices: Optional[int] = None,
                 add_water: bool = True,):
        self._resolution = resolution
        self._pixel_size = 0.5 * resolution
        self.microscope = microscope
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

    def run(self, mol):
        wave_propagator_t = wave.get_wave_propagator()
        slice_builder = dens.get_slice_builder()
        wave_propagator = wave_propagator_t(self.roi, self._pixel_size, self.microscope.beam_energy_kev)
        mol = atm.centralize(mol)
        slices = slice_builder(mol,
                               pixel_size=self._pixel_size,
                               dz=self.slice_thickness,
                               lateral_size=self.roi,
                               add_water=self.add_water)

        init_wave = wave_propagator.init_wave(self.microscope.electron_dose)

        exit_wave = wave_propagator.multislice_propagate(init_wave, slices, self.slice_thickness)

        image_wave = wave_propagator.lens_propagate(exit_wave,
                                                    self.microscope.cs_mm,
                                                    self.microscope.defocus,
                                                    self.microscope.aperture)

        image = image_wave.real ** 2 + image_wave.imag ** 2

        return image

    def __repr__(self):
        return f"{{ImagePipe | resolution = {self._resolution}Angstrom, " \
               f"slice_thickness = {self.slice_thickness}Angstrom, " \
               f"roi = {self.roi}, " \
               f"add_water = {self.add_water}, " \
               f"microscope = {self.microscope}}}"

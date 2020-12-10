from typing import Tuple, Optional, Union

from . import pot
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
                 add_water: bool = True,
                 upto_exit_wave: bool = False):
        self._resolution = resolution
        self._pixel_size = 0.5 * resolution
        self.microscope = microscope
        self.slice_thickness = slice_thickness
        self.add_water = add_water
        self.upto_exit_wave = upto_exit_wave

        if type(roi) is int:
            self.roi = (roi, roi)
        else:
            self.roi = roi
        self.n_slices = n_slices

        self.wave_propagator = wave.get_wave_propagator(self.roi, self._pixel_size, self.microscope.beam_energy_kev)


    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        self._pixel_size = 0.5 * res
        self._resolution = res

    # get the exit wave (after specimen, before lens)
    def get_exit_wave(self, mol):
        mol = atm.centralize(mol)
        slices = pot.build_multi_slices(mol,
                                        pixel_size=self._pixel_size,
                                        dz=self.slice_thickness,
                                        lateral_size=self.roi,
                                        add_water=self.add_water)

        init_wave = self.wave_propagator.init_wave(self.microscope.electron_dose)
        exit_wave = self.wave_propagator.multislice_propagate(init_wave, slices, self.slice_thickness)
        return exit_wave

    def get_exit_wave_wpo(self, mol):
        mol = atm.centralize(mol)
        aslice = pot.build_one_slice(mol, 
                                     pixel_size=self._pixel_size,
                                     lateral_size=self.roi)
        init_wave = self.wave_propagator.init_wave(self.microscope.electron_dose)
        exit_wave = self.wave_propagator.singleslice_propagate(init_wave, aslice, dz=0)
        return exit_wave

    def lens_propagate(self, exit_wave):
        return self.wave_propagator.lens_propagate(exit_wave,
                                                   self.microscope.cs_mm,
                                                   self.microscope.defocus,
                                                   self.microscope.aperture)

    # directly get final image (real valued)

    def run_wpo(self, mol):
        exit_wave = self.get_exit_wave_wpo(mol)
        image_wave = self.lens_propagate(exit_wave)
        image = image_wave.real ** 2 + image_wave.imag ** 2
        return image

    def run(self, mol):
        exit_wave = self.get_exit_wave(mol)
        image_wave = self.lens_propagate(exit_wave)
        image = image_wave.real ** 2 + image_wave.imag ** 2
        return image

    def __repr__(self):
        return f"{{ImagePipe | resolution = {self._resolution}Angstrom, " \
               f"slice_thickness = {self.slice_thickness}Angstrom, " \
               f"roi = {self.roi}, " \
               f"add_water = {self.add_water}, " \
               f"microscope = {self.microscope}}}"

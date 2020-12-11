from typing import Tuple, Union

from . import config


# dynamically get the backend wave propagator
def get_raw_wave_propagator(wave_shape: Union[int, Tuple[int, int]], pixel_size: Union[float, Tuple[float, float]], beam_energy_kev: float):
    
    if type(wave_shape) is int:
        wave_shape = (wave_shape, )*2
    if type(pixel_size) is float:
        pixel_size = (pixel_size, )*2
    
    backend = config.get_current_backend()
    wave_propagator = backend.wave_propagator(wave_shape[0], wave_shape[1], pixel_size[0], pixel_size[1], beam_energy_kev)
    return wave_propagator


# the frontend propagator
class WavePropagator:
    def __init__(self, wave_shape: Union[int, Tuple[int, int]], pixel_size: Union[float, Tuple[float, float]],
                 beam_energy_kev: float, electron_dose: float, cs_mm: float, defocus: float, aperture: float):

        self.electron_dose = electron_dose
        self.cs_mm = cs_mm
        self.defocus = defocus
        self.aperture = aperture
        self.raw_propagator = get_raw_wave_propagator(wave_shape, pixel_size, beam_energy_kev)

    def init_wave(self):
        return self.raw_propagator.init_wave(self.electron_dose)
    
    def slice_transmit(self, wave, aslice):
        return self.raw_propagator.slice_transmit(wave, aslice)

    def space_propagate(self, wave, dz):
        return self.raw_propagator.space_propagate(wave, dz)

    def multislice_propagate(self, wave, slices, dz: float):
        return self.raw_propagator.multislice_propagate(wave, slices, dz)

    def singleslice_propagate(self, wave, aslice, dz: float):
        return self.raw_propagator.singleslice_propagate(wave, aslice, dz)

    def lens_propagate(self, wave):
        return self.raw_propagator.lens_propagate(wave, self.cs_mm, self.defocus, self.aperture)

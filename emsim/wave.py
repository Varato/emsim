from typing import Tuple

from . import config


def get_wave_propagator(shape: Tuple[int, int], pixel_size: float, beam_energy_kev: float):
    backend = config.get_current_backend()
    wave_propagator = backend.wave_propagator(shape, pixel_size, beam_energy_kev)
    return wave_propagator



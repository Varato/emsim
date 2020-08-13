from . import config


def get_wave_propagator():
    backend = config.get_current_backend()
    wave_propagator = backend.wave_propagator
    return wave_propagator



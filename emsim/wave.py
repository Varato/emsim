from typing import Union, Optional, Tuple
import numpy as np
from numpy.fft import fft2, rfft2, irfft2, ifftshift

from . import config
from . import back_end
from .back_end import requires_cuda_ext, requires_c_ext
from . import atoms as atm
from . import elem
from .physics import a0, e

float_type = np.float32


def get_wave_propagator():
    backend = config.get_current_backend()
    wave_propagator = backend.wave_propagator
    return wave_propagator



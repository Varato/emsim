import math
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import PIL
from PIL import Image

import emsim


def assure_numpy_array(arr):
    if type(arr) is np.ndarray:
        return arr
    return arr.get()


def make_intensity(wave):
    if type(wave) is cp.ndarray:
        return np.abs(wave.get())**2
    else:
        return np.abs(wave)**2


microscope = emsim.em.EM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=math.pi/36.
)

mol = emsim.atoms.AtomList(
    elements=np.array([6, 8], dtype=np.int), 
    coordinates=np.array([[0,-20,-20], [0, 20, 20]], dtype=np.float32))

emsim.config.set_backend('numpy')
image_shape_ = (128, 128)
pixel_size_ = 1.5

sb = emsim.dens.get_slice_builder()
wp = emsim.wave.get_wave_propagator(
    shape=image_shape_, pixel_size=pixel_size_, beam_energy_kev=microscope.beam_energy_kev)

# ImageOps.grayscale(og_image)
curr_path = Path(__file__).parent.resolve()
# print(curr_path)
letterA = np.load(curr_path/"images/LetterA.npy")

aslice = letterA.astype(np.float32) # np.roll(letterA, shift=[0,-35], axis=(-2, -1)) + np.roll(letterA, shift=[0, 35], axis=(-2, -1)) * 2
aslice = np.ascontiguousarray(aslice)
# aslice = sb(mol, dz=3., pixel_size=pixel_size_, lateral_size=image_shape_, n_slices=1)[0]
wave = wp.init_wave(microscope.electron_dose)
wave = wp.slice_transmit(wave, aslice)
# wave = wp.singleslice_propagate(wave, aslice, dz=0.1)
wave = wp.space_propagate(wave, 1)
wave = wp.lens_propagate(wave, microscope.cs_mm, microscope.defocus, microscope.aperture)

_, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(assure_numpy_array(aslice), cmap="Greys")
ax2.imshow(make_intensity(wave), cmap="Greys")
plt.show()
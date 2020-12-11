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


tem = emsim.tem.TEM(
    electron_dose=100,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=10.37e-3
)

mol = emsim.atoms.AtomList(
    elements=np.array([6, 14, 29, 79, 92], dtype=np.int), 
    coordinates=np.array([[0,0,5], [0,0,15], [0,0,25],[0,0,35], [0,0,45]], dtype=np.float32))

mol = emsim.atoms.centralize(mol)

emsim.config.set_backend('numpy')
image_shape_ = (512, 512)
pixel_size_ = 50/512

propagator = tem.get_wave_propagator(wave_shape=image_shape_, pixel_size=pixel_size_)
aslice = emsim.pot.build_one_slice(mol, pixel_size=pixel_size_, lateral_size=image_shape_)

wave = propagator.init_wave()
wave = propagator.slice_transmit(wave, aslice)
line = assure_numpy_array(wave[512//2,:])
wave = propagator.lens_propagate(wave)

image = make_intensity(wave)
image_line = image[512//2,:]

fig1, ax1 = plt.subplots()
ax1.imshow(image, cmap="gray")

xx = np.arange(512) * pixel_size_
fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.plot(xx, line.real)
ax2.plot(xx, line.imag)
ax3.plot(xx, image_line)
ax1.set_ylabel("real part")
ax2.set_ylabel("imag part")
ax3.set_ylabel("intensity")
plt.show()
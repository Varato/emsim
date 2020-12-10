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

def symmetric_band_limit(arr):
    m, n = arr.shape
    r = min(m,n) // 2
    kx, ky = np.meshgrid(np.arange(-m//2, -m//2 + m), np.arange(-n//2, -n//2 + m))
    k2 = kx**2 + ky**2
    fil = np.where(k2 <= r**2, 1, 0)
    fil = np.fft.ifftshift(fil)
    return np.fft.ifft2(np.fft.fft2(arr) * fil)


microscope = emsim.em.EM(
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


emsim.config.set_backend('cuda')
image_shape_ = (512, 512)
pixel_size_ = 50/512


sb = emsim.dens.get_single_slice_builder()
wp = emsim.wave.get_wave_propagator(
    shape=image_shape_, pixel_size=pixel_size_, beam_energy_kev=microscope.beam_energy_kev)

aslice = sb(mol, pixel_size=pixel_size_, lateral_size=image_shape_)

wave = wp.init_wave(microscope.electron_dose)
wave = wp.slice_transmit(wave, aslice)
# wave = symmetric_band_limit(wave)
line = assure_numpy_array(wave[512//2,:])
wave = wp.lens_propagate(wave, microscope.cs_mm, microscope.defocus, microscope.aperture)

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
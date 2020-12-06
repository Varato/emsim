import math
import numpy as np
import matplotlib.pyplot as plt
import time
import PIL
from PIL import Image
import emsim

def make_intensity(wave):
    return np.abs(wave)**2

microscope = emsim.em.EM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=math.pi/36.
)

mol = emsim.atoms.AtomList(
    elements=np.array([6], dtype=np.int), 
    coordinates=np.array([[0,0,0]], dtype=np.float32))
emsim.config.set_backend('cuda')
image_shape_ = (128, 128)
pixel_size_ = 1.5

sb = emsim.dens.get_single_slice_builder()
wp = emsim.wave.get_wave_propagator(
    shape=image_shape_, pixel_size=pixel_size_, beam_energy_kev=microscope.beam_energy_kev)

# ImageOps.grayscale(og_image)
letterA = np.load("./images/LetterA.npy")

aslice = np.roll(letterA, shift=[0,-35], axis=(-2, -1)) + np.roll(letterA, shift=[0, 35], axis=(-2, -1)) * np.pi*2

# aslice = sb(mol, pixel_size=pixel_size_, lateral_size=image_shape_)

init_wave = wp.init_wave(microscope.electron_dose)
exit_wave = wp.slice_transmit(init_wave, aslice)
fresnel_wave = wp.space_propagate(exit_wave, 0.1)
image_wave = wp.lens_propagate(fresnel_wave, microscope.cs_mm, microscope.defocus, microscope.aperture)

plt.imshow(make_intensity(image_wave))
plt.show()
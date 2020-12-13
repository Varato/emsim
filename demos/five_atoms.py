import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import emsim


tem = emsim.tem.TEM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=10.37e-3
)

tem.set_scherzer_condition()

# The example from Kirkland 2010, P.95
# Five atoms, C, Si, Cu, Au, U, are placed on the z=0 plane, with coordinates (z=0, x, y) in unit of Angstroms.
mol = emsim.atoms.AtomList(
    elements=np.array([6, 14, 29, 79, 92], dtype=np.int), 
    coordinates=np.array([[0,0,5], [0,0,15], [0,0,25],[0,0,35], [0,0,45]], dtype=np.float32))
mol = emsim.atoms.centralize(mol)

# set backend. Choices are numpy, fftw, cuda
emsim.config.set_backend('cuda')
image_shape_ = (512, 512)
pixel_size_ = 50/512

propagator = tem.get_wave_propagator(wave_shape=image_shape_, pixel_size=pixel_size_)
aslice = emsim.pot.build_one_slice(mol, pixel_size=pixel_size_, lateral_size=image_shape_)

wave = propagator.init_wave()                  
exit_wave = propagator.slice_transmit(wave, aslice)
image_wave = propagator.lens_propagate(exit_wave)

image = emsim.utils.array.assure_numpy(abs(image_wave)**2)

wave_line = emsim.utils.array.assure_numpy(exit_wave[512//2,:])
image_line = emsim.utils.array.assure_numpy(image[512//2,:])
fig1, ax1 = plt.subplots()
ax1.imshow(image, cmap="gray")

xx = np.arange(512) * pixel_size_
fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.plot(xx, wave_line.real)
ax2.plot(xx, wave_line.imag)
ax3.plot(xx, image_line)
ax1.set_ylabel("real part")
ax2.set_ylabel("imag part")
ax3.set_ylabel("intensity")

cur_path = Path(__file__).resolve().parent
fig1.savefig(cur_path / 'images/fiveAtoms.png', dpi=120, facecolor='none')
fig2.savefig(cur_path / 'images/fiveAtomsLines.png', dpi=120, facecolor='none')
plt.show()
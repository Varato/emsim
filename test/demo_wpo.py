import math
import numpy as np
import matplotlib.pyplot as plt
import time

import emsim

def assure_numpy_array(arr):
    if type(arr) is np.ndarray:
        return arr
    return arr.get()

microscope = emsim.em.EM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=10.3
)

image_pipe = emsim.pipe.Pipe(
    microscope=microscope,
    resolution=512/50*2,
    slice_thickness=2,
    roi=512,
    add_water=False,
)

emsim.config.set_backend('numpy')

mol = emsim.atoms.AtomList(
    elements=np.array([6, 14, 29, 79, 92], dtype=np.int), 
    coordinates=np.array([[0,0,5], [0,0,15], [0,0,25],[0,0,35], [0,0,45]], dtype=np.float32))

img = image_pipe.run(mol)
plt.imshow(assure_numpy_array(img), cmap='gray')
plt.show()
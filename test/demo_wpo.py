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
    defocus=670,
    aperture=math.pi/2
)

image_pipe = emsim.pipe.Pipe(
    microscope=microscope,
    resolution=3,
    slice_thickness=2,
    roi=256,
    add_water=True,
)

emsim.config.set_backend('cuda')

mol = emsim.atoms.AtomList(
    elements=np.array([6], dtype=np.int), 
    coordinates=np.array([[0,0,0]], dtype=np.float32))

img = image_pipe.run_wpo(mol)
plt.imshow(assure_numpy_array(img), cmap='Greys')
plt.show()
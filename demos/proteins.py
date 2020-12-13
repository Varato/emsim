import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

import emsim


emsim.config.set_backend("cuda")


class Molecules(object):
    def __init__(self):
        self.pdbs = ["7ahl", "5l54", "4ear", "1fat"]

    def __iter__(self):
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        for pdb_code in self.pdbs:
            pdb_file = emsim.utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
            mol = emsim.utils.pdb.build_biological_unit(pdb_file)
            mol.label = pdb_code
            # get an orientation for each molecule
            quat = np.array([1., 0., 0., 0.]) # emsim.utils.rot.random_uniform_quaternions(1)
            yield emsim.atoms.rotate(mol, quat, set_center=True)


class ResultHandler:
    def __init__(self):
        self.images = []

    def __call__(self, result, label):
        print(f"got image for label = {label}")
        self.images.append(result)


microscope = emsim.tem.TEM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=10e-3
)

image_pipe = emsim.pipe.Pipe(
    microscope=microscope,
    resolution=2,
    slice_thickness=2,
    roi=256,
    add_water=False,
)


print(str(image_pipe))

result_handler = ResultHandler()
mols = Molecules()
sim = emsim.simulator.EMSim(image_pipe, mols, result_handler)
start = time.time()
sim.run()
time_elapsed = time.time() - start

print(f"time elaplsed: {time_elapsed:.4f}")

fig, axes = plt.subplots(ncols=len(result_handler.images), figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(result_handler.images[i], cmap="gray")
    ax.set_title(mols.pdbs[i])
    ax.axis(False)

fig.subplots_adjust(wspace=0)
cur_path = Path(__file__).resolve().parent
fig.savefig(cur_path / 'images/proteins.png', dpi=120, facecolor='none')
plt.show()


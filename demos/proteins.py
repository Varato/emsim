import math
import matplotlib.pyplot as plt
import time

import emsim


class Molecules(object):
    def __init__(self):
        self.pdbs = ["7ahl", "4bed", "4ear", "1fat"]

    def __iter__(self):
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        for pdb_code in self.pdbs:
            pdb_file = emsim.utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
            mol = emsim.utils.pdb.build_biological_unit(pdb_file)
            mol.label = pdb_code
            quat = emsim.utils.rot.random_uniform_quaternions(1)
            yield emsim.atoms.rotate(mol, quat, set_center=True)


class ResultHandler:
    def __init__(self):
        self.images = []

    def __call__(self, result, label):
        print(f"got image for label = {label}")
        self.images.append(result)


microscope = emsim.em.EM(
    electron_dose=20,
    beam_energy_kev=200,
    cs_mm=1.3,
    defocus=700,
    aperture=10.72e-3
)

image_pipe = emsim.pipe.Pipe(
    microscope=microscope,
    resolution=3,
    slice_thickness=2,
    roi=256,
    add_water=False,
)


if __name__ == "__main__":
    print(str(image_pipe))
    emsim.config.set_backend("cuda")

    result_handler = ResultHandler()
    sim = emsim.simulator.EMSim(image_pipe, Molecules(), result_handler)
    start = time.time()
    sim.run()
    time_elapsed = time.time() - start

    print(f"time elaplsed: {time_elapsed:.4f}")


    _, axes = plt.subplots(ncols=len(result_handler.images))
    for i, ax in enumerate(axes):
        ax.imshow(result_handler.images[i], cmap="gray")
    plt.show()


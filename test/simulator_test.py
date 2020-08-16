import math
import matplotlib.pyplot as plt

import emsim


microscope = emsim.em.EM(
    electron_dose=20,
    beam_energy_kev=300,
    cs_mm=1.3,
    defocus=700,
    aperture=math.pi/2
)

image_pipe = emsim.pipe.Pipe(
    microscope=microscope, 
    resolution=2, 
    slice_thickness=2, 
    roi=128,
    add_water=False, 
)

class Molecules(object):
    def __init__(self):
        self.pdbs = ["7ahl", "4ear"]

    def __iter__(self):
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        for pdb_code in self.pdbs:
            pdb_file = emsim.utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
            mol = emsim.utils.pdb.build_biological_unit(pdb_file)
            yield mol

def result_handler(result):
    global images
    print(f"result received: image {result.shape}")
    images.append(result)
    
images = []

if __name__ == "__main__":
    sim = emsim.simulator.EMSim(image_pipe, Molecules(), result_handler)
    sim.run()

    _, axes = plt.subplots(ncols=len(images))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
    plt.show()


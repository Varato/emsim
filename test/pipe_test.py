import unittest
import numpy as np
import matplotlib.pyplot as plt
import time

import emsim
from emsim import em
from emsim import utils
from emsim import pipe


class PipeTestCase(unittest.TestCase):
    def setUp(self):
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        pdb_code = '7ahl'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
        self.mol = utils.pdb.build_biological_unit(pdb_file)

        self.resolution = 3.0
        beam_energy_kev = 200
        cs = 2.0  # mm
        defocus = 700  # Angstrom
        dose = 20
        aperture = np.pi/2.0
        self.thickness = 2.0
        self.microscope = em.EM(dose, beam_energy_kev, cs, defocus, aperture)

    def test_image_numpy(self):
        emsim.config.set_backend("numpy")
        p = pipe.Pipe(self.microscope, self.resolution, self.thickness, add_water=True, roi=128)
        img = p.run(self.mol)
        _, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.show()

    def test_image_fftw(self):
        emsim.config.set_backend("fftw")
        p = pipe.Pipe(self.microscope, self.resolution, self.thickness, add_water=False, roi=128)
        img = p.run(self.mol)
        _, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.show()

    def test_image_cuda(self):
        emsim.config.set_backend("cuda")
        p = pipe.Pipe(self.microscope, self.resolution, self.thickness, add_water=False, roi=128)
        img = p.run(self.mol)
        _, ax = plt.subplots()
        ax.imshow(img.get(), cmap='gray')
        plt.show()

    def test_image_timing(self):
        p = pipe.Pipe(self.microscope, self.resolution, self.thickness, add_water=False, roi=256)
        imgs = []
        t0 = time.time()
        p.set_backend("numpy")
        imgs.append(p.run(self.mol))
        t1 = time.time()
        p.set_backend("fftw")
        imgs.append(p.run(self.mol))
        t2 = time.time()
        p.set_backend("cuda")
        imgs.append(p.run(self.mol).get())
        t3 = time.time()

        print(f"numpy time = {t1-t0:.3f}")
        print(f"fftw time = {t2-t1:.3f}")
        print(f"cuda time = {t3-t2:.3f}")

        print(f"mse numpy fftw: {np.mean((imgs[0] - imgs[1])**2)}")
        print(f"mse numpy cuda: {np.mean((imgs[0] - imgs[2])**2)}")
        print(f"mse fftw  cuda: {np.mean((imgs[1] - imgs[2])**2)}")

        _, axes = plt.subplots(ncols=4)
        for i in range(3):
            axes[i].imshow(imgs[i], cmap="gray")
        axes[3].imshow(imgs[0] - imgs[2])
        plt.show()


if __name__ == '__main__':
    unittest.main()

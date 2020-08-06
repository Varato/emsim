import unittest
import numpy as np
import matplotlib.pyplot as plt
import time

import emsim
from emsim import em
from emsim import utils
from emsim import pipe


class MultislicePipeTestCase(unittest.TestCase):
    def setUp(self):
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        pdb_code = '4bed'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
        self.mol = utils.pdb.build_biological_unit(pdb_file)

        resolution = 3.0
        beam_energy_kev = 200
        cs = 2.0  # mm
        defocus = 700  # Angstrom
        dose = 20
        aperture = np.pi/2.0
        thickness = 1.0

        self.microscope = em.EM(dose, beam_energy_kev, cs, defocus, aperture)
        self.pipe = pipe.MultislicePipe(self.microscope, self.mol, resolution, thickness,
                                        add_water=False,
                                        roi=256)

    def test_image_numpy(self):
        img = self.pipe.run(back_end='numpy')
        _, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.show()

    def test_image_fftw(self):
        img = self.pipe.run(back_end='fftw')
        _, ax = plt.subplots()
        ax.imshow(np.abs(img), cmap='gray')
        plt.show()

    def test_image_cuda(self):
        img = self.pipe.run(back_end='cuda').get()

        _, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        plt.show()

    def test_image_timing(self):
        imgs = []
        t0 = time.time()
        imgs.append(self.pipe.run(back_end='numpy'))
        t1 = time.time()
        imgs.append(self.pipe.run(back_end='fftw'))
        t2 = time.time()
        imgs.append(self.pipe.run(back_end='cuda').get())
        t3 = time.time()

        print(f"numpy time = {t1-t0:.3f}")
        print(f"fftw time = {t2-t1:.3f}")
        print(f"cuda time = {t3-t2:.3f}")

        print(f"diff numpy fftw: {np.abs(imgs[0] - imgs[1]).max()}")
        print(f"diff numpy cuda: {np.abs(imgs[0] - imgs[2]).max()}")
        print(f"diff fftw  cuda: {np.abs(imgs[1] - imgs[2]).max()}")

        _, axes = plt.subplots(ncols=3)
        for i in range(3):
            axes[i].imshow(imgs[i])
        plt.show()


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import matplotlib.pyplot as plt
import time

import emsim
from emsim import em
from emsim import utils
from emsim import atoms as atm
from emsim import elem
from emsim import dens


class EmImagingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        pdb_code = '7ahl'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
        mol = utils.pdb.build_biological_unit(pdb_file)

        resolution = 2.0
        beam_energy_kev = 200
        cs = 2.0  # mm
        defocus = 700  # Angstrom
        dose = 20
        thickness = 3.0
        pixel_size = resolution / 2.0

        slcs = dens.build_slices_fourier_fftw(mol, pixel_size, thickness, lateral_size=256, add_water=False)
        self.specimen = em.Specimen(slcs, pixel_size, thickness)
        self.em = em.EM(dose, beam_energy_kev, cs, defocus, aperture=np.pi/2.0)

    def test_em_imaging_fftw(self):
        psi = self.em.make_image(self.specimen, kernel="fftw")
        print("final")
        print(psi.flags)

    def test_em_imaging_np(self):
        psi = self.em.make_image(self.specimen, kernel="np")
        print("final")
        print(psi.flags)

    def test_em_imaging_consistency(self):
        t0 = time.time()
        psi1 = self.em.make_image(self.specimen, kernel="np")
        t1 = time.time()
        psi2 = self.em.make_image(self.specimen, kernel="fftw")
        t2 = time.time()
        img1, img2 = np.abs(psi1)**2, np.abs(psi2)**2
        print(f"max difference = {(np.abs(img1-img2)).max():.3f}")
        print(f"max relative difference = {((np.abs(img1-img2))/img1).max():.3f}")
        print(f"np time: {t1-t0:.3f}, fftw time: {t2-t1:.3f}")

        _, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 3))
        ax1.imshow(img1, cmap="gray")
        ax2.imshow(img2, cmap="gray")
        ax3.imshow(np.abs(img1-img2))
        plt.show()


if __name__ == '__main__':
    unittest.main()

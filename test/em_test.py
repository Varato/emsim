import unittest
import numpy as np
import matplotlib.pyplot as plt

import emsim
from emsim import em
from emsim import utils
from emsim import atoms as atm
from emsim import elem
from emsim import dens


class EmImagingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pdb_data_dir = emsim.io.data_dir.get_pdb_data_dir_from_config()
        pdb_code = '4ear'
        pdb_file = utils.pdb.retrieve_pdb_file(pdb_code, pdb_data_dir)
        mol = utils.pdb.build_biological_unit(pdb_file)

        resolution = 1.0
        beam_energy_kev = 200
        cs = 2.0  # mm
        defocus = 700  # Angstrom
        dose = 20
        thickness = 3.0
        pixel_size = resolution / 2.0

        slcs = dens.build_slices_fourier(mol, pixel_size, thickness, lateral_size=256, add_water=False)
        self.specimen = em.Specimen(slcs, pixel_size, thickness)
        self.em = em.EM(dose, beam_energy_kev, cs, defocus, aperture=np.pi/2.0)

    def test_em_imaging(self):
        psi = self.em.make_image(self.specimen)
        plt.imshow(np.abs(psi)**2, cmap='gray')
        plt.show()


if __name__ == '__main__':
    unittest.main()

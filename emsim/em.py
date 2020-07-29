
from typing import Tuple
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2

from .physics import electron_wave_length_angstrom, mtf, aberration, electron_relativity_gamma


class Specimen(object):
    def __init__(self, slices, pixel_size, dz):
        self.slices = slices
        self.pixel_size = pixel_size
        self.dz = dz

    @property
    def lateral_size(self):
        return self.slices.shape[1:]

    @property
    def num_slices(self):
        return self.slices.shape[0]

    def __iter__(self):
        self._s = 0
        return self

    def __next__(self):
        if self._s < self.num_slices:
            slc = self.slices[self._s, ...]
            self._s += 1
            return slc
        else:
            raise StopIteration


def band_limit_specimen(specimen: Specimen) -> Specimen:
    s1, s2 = specimen.lateral_size
    r = min([s1, s2]) // 2
    x, y = np.mgrid[0:s1, 0:s2]
    fil = ifftshift(np.where((x - s1 // 2) ** 2 + (y - s2 // 2) ** 2 <= 4 * r * r / 9, 1., 0.))
    filtered_slices = ifft2(fft2(specimen.slices, axes=(0, 1)) * fil[..., None], axes=(0, 1)).real
    return Specimen(filtered_slices, specimen.pixel_size, specimen.dz)


class EM(object):
    def __init__(self, electron_dose: float, beam_energy_kev: float, cs: float, defocus: float, aperture: float):
        self.electron_dose = electron_dose
        self.beam_energy_kev = beam_energy_kev
        self.cs = cs
        self.defocus = defocus
        self.aperture = aperture
        self.relativity_gamma = electron_relativity_gamma(beam_energy_kev)

        self.wave_length_angstrom = electron_wave_length_angstrom(beam_energy_kev)

        self.aberr_ = aberration(self.wave_length_angstrom, cs, defocus)
        self.mtf_ = mtf(self.wave_length_angstrom, cs, defocus)

    def make_wave_in(self, pixel_size: float, lateral_size: Tuple[int, int]):
        n_e = self.electron_dose * pixel_size ** 2
        wave_in = np.ones(lateral_size, dtype=np.complex64)
        wave_in *= np.sqrt(n_e) / np.abs(wave_in)
        return wave_in

    def make_image(self, specimen: Specimen, kernel="fftw"):
        wave_in = self.make_wave_in(specimen.pixel_size, specimen.lateral_size)
        qx, qy = EM._make_mesh_grid_fourier_space(specimen.pixel_size, specimen.lateral_size)
        q_mgrid = np.sqrt(qx*qx + qy*qy)

        if kernel == "fftw":
            psi = self.multislice_propagate_fftw(specimen, wave_in)
        else:
            psi = self.multislice_propagate_np(specimen, wave_in, q_mgrid)
        return self.lens_propagate(psi, q_mgrid)

    def multislice_propagate_fftw(self, specimen: Specimen, wave_in: np.ndarray):
        from .ext import em_kernel
        return em_kernel.multislice_propagate_fftw(
            wave_in, specimen.slices.astype(np.float32),
            specimen.pixel_size, specimen.dz, self.wave_length_angstrom, self.relativity_gamma)

    def multislice_propagate_np(self, specimen: Specimen, wave_in: np.ndarray, q_mgrid: np.ndarray):
        q_max = 0.5 / specimen.pixel_size
        fil = np.fft.ifftshift(np.where(q_mgrid <= q_max*0.6667, 1., 0.))
        spatial_propagator = np.exp(-1j * self.wave_length_angstrom * np.pi * specimen.dz * q_mgrid**2)
        transmission_functions = np.exp(1j * self.relativity_gamma * self.wave_length_angstrom * specimen.slices)
        psi = wave_in
        for s in range(specimen.num_slices):
            psi = ifft2(ifftshift(spatial_propagator) * fft2(psi * transmission_functions[s]) * fil)
        return psi

    def projection_approx_propagate(self, specimen: Specimen, wave_in: np.ndarray, q_mgrid: np.ndarray):
        vz = specimen.slices.sum(-1)
        t = np.exp(1j * self.relativity_gamma * self.wave_length_angstrom * vz)
        return wave_in * t

    def lens_propagate(self, wave_in: np.ndarray, q_mgrid: np.ndarray):
        h = self.mtf_(q_mgrid)
        apper = np.where(q_mgrid < self.aperture / self.wave_length_angstrom, 1., 0.)
        wave_out = ifft2(ifftshift(h) * fft2(wave_in) * ifftshift(apper))
        print("wave_in")
        print(wave_in.flags)
        print("h")
        print(h.flags)
        print("apper")
        print(apper.flags)
        return wave_out

    @staticmethod
    def _make_mesh_grid_fourier_space(pixel_size: float, size: Tuple[int, int]):
        q_max = 0.5/pixel_size
        qx_range = np.linspace(-q_max, q_max, size[0])
        qy_range = np.linspace(-q_max, q_max, size[1])
        qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
        return qx, qy

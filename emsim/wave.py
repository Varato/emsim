from typing import Tuple, Union
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


from .physics import mtf
from . import back_end
from .back_end import requires_c_ext, requires_cuda_ext
from . import back_end


def init_wave(electron_dose: float, pixel_size: float, lateral_size: Union[int, Tuple[int, int]]):
    if type(lateral_size) is int:
        lateral_size = (lateral_size, lateral_size)
    n_e = electron_dose * pixel_size ** 2
    wave_in = np.ones(lateral_size, dtype=np.complex64)
    wave_in *= np.sqrt(n_e) / np.abs(wave_in)
    return wave_in


@requires_cuda_ext
def init_wave_cuda(electron_dose: float, pixel_size: float, lateral_size: Union[int, Tuple[int, int]]):
    if type(lateral_size) is int:
        lateral_size = (lateral_size, lateral_size)
    cp = back_end.cp
    n_e = electron_dose * pixel_size ** 2
    wave = cp.ones(lateral_size, dtype=np.complex64)
    wave *= cp.sqrt(n_e) / np.abs(wave)
    return wave


def multislice_propagate(wave_in: np.ndarray,
                         slices: np.ndarray, pixel_size: float, dz: float,
                         wave_length: float, relativity_gamma: float):
    n_slices = slices.shape[0]
    lateral_size = wave_in.shape
    qx, qy = _make_mesh_grid_fourier_space(pixel_size, lateral_size)
    q_mgrid = np.sqrt(qx * qx + qy * qy)
    q_max = 0.5 / pixel_size

    fil = np.fft.ifftshift(np.where(q_mgrid <= q_max * 0.6667, 1., 0.))
    spatial_propagator = np.exp(-1j * wave_length * np.pi * dz * q_mgrid ** 2)
    transmission_functions = np.exp(1j * relativity_gamma * wave_length * slices)

    psi = wave_in
    for s in range(n_slices):
        psi = ifft2(ifftshift(spatial_propagator) * fft2(psi * transmission_functions[s]) * fil)
    return psi


def lens_propagate(wave_in: np.ndarray, pixel_size: float,
                   wave_length: float, cs_mm: float, defocus: float, aperture: float):
    lateral_size = wave_in.shape
    qx, qy = _make_mesh_grid_fourier_space(pixel_size, lateral_size)
    q_mgrid = np.sqrt(qx * qx + qy * qy)

    h = mtf(wave_length, cs_mm, defocus)(q_mgrid)
    aper = np.where(q_mgrid < aperture / wave_length, 1., 0.)
    return ifft2(ifftshift(h) * fft2(wave_in) * ifftshift(aper))


@requires_c_ext
def multislice_propagate_fftw(wave_in: np.ndarray,
                              slices: np.ndarray, pixel_size: float, dz: float,
                              wave_length: float, relativity_gamma: float):
    return back_end.em_kernel.multislice_propagate_fftw(
        wave_in, slices.astype(np.float32),
        pixel_size, dz, wave_length, relativity_gamma)


@requires_c_ext
def lens_propagate_fftw(wave_in: np.ndarray, pixel_size: float,
                        wave_length: float, cs_mm: float, defocus: float, aperture: float):
    return back_end.em_kernel.lens_propagate_fftw(
        wave_in, pixel_size,
        wave_length, cs_mm, defocus, aperture)


# @requires_cuda_ext
# def multislice_propagate_cuda(wave_in: back_end.cp.ndarray,
#                               slices: back_end.cp.ndarray, pixel_size: float, dz: float,
#                               wave_length: float, relativity_gamma: float):
#     return back_end.em_kernel_cuda.multislice_propagate_cuda(
#         wave_in, slices,
#         pixel_size, dz, wave_length, relativity_gamma)
#
#
# @requires_cuda_ext
# def lens_propagate_cuda(wave_in: back_end.cp.ndarray, pixel_size: float,
#                         wave_length: float, cs_mm: float, defocus: float, aperture: float):
#     return back_end.em_kernel_cuda.lens_propagate_cuda(
#         wave_in, pixel_size,
#         wave_length, cs_mm, defocus, aperture)

class WavePropagator:
    def __init__(self, wave_shape: Tuple[int, int], pixel_size, wave_length, relativity_gamma,
                 backend=back_end.em_kernel_cuda.WavePropagator):
        self.wave_shape = wave_shape
        self.pixel_size = pixel_size
        self.wave_length = wave_length
        self.relativity_gamma = relativity_gamma
        if backend == "numpy":
            self.backend = _WavePropagatorNumpy(wave_shape[0], wave_shape[1], pixel_size, wave_length, relativity_gamma)
        else:
            self.backend = backend(wave_shape[0], wave_shape[1], pixel_size, wave_length, relativity_gamma)

    def multislice_propagate(self, wave_in, slices, dz: float):
        return self.backend.multislice_propagate(wave_in, slices, dz)

    def singleslice_propagate(self, wave_in, aslice, dz: float):
        return self.backend.singleslice_propagate(wave_in, aslice, dz)

    def lens_propagate(self, wave_in, cs_mm, defocus, aperture):
        return self.backend.lens_propagate(wave_in, cs_mm, defocus, aperture)


class _WavePropagatorNumpy:
    def __init__(self, n1, n2, pixel_size, wave_length, relativity_gamma):
        self.wave_shape = (n1, n2)
        self.pixel_size = pixel_size
        self.wave_length = wave_length
        self.relativity_gamma = relativity_gamma

        qx, qy = _WavePropagatorNumpy._make_mesh_grid_fourier_space(pixel_size, self.wave_shape)
        self.q_mgrid = np.sqrt(qx * qx + qy * qy)

    def multislice_propagate(self, wave_in, slices, dz: float):
        n_slices = slices.shape[0]
        q_max = 0.5 / self.pixel_size

        fil = np.fft.ifftshift(np.where(self.q_mgrid <= q_max * 0.6667, 1., 0.))
        spatial_propagator = np.exp(-1j * self.wave_length * np.pi * dz * self.q_mgrid ** 2)
        transmission_functions = np.exp(1j * self.relativity_gamma * self.wave_length * slices)

        psi = wave_in
        for s in range(n_slices):
            psi = ifft2(ifftshift(spatial_propagator) * fft2(psi * transmission_functions[s]) * fil)
        return psi

    def singleslice_propagate(self, wave_in, aslice, dz: float):
        raise NotImplemented

    def lens_propagate(self, wave_in, cs_mm, defocus, aperture):
        h = mtf(self.wave_length, cs_mm, defocus)(self.q_mgrid)
        aper = np.where(self.q_mgrid < aperture / self.wave_length, 1., 0.)
        return ifft2(ifftshift(h) * fft2(wave_in) * ifftshift(aper))

    @staticmethod
    def _make_mesh_grid_fourier_space(pixel_size: float, size: Tuple[int, int]):
        q_max = 0.5/pixel_size
        qx_range = np.linspace(-q_max, q_max, size[0])
        qy_range = np.linspace(-q_max, q_max, size[1])
        qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
        return qx, qy

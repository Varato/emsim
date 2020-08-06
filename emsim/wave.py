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


@requires_cuda_ext
def multislice_propagate_cuda(wave_in: back_end.cp.ndarray,
                              slices: back_end.cp.ndarray, pixel_size: float, dz: float,
                              wave_length: float, relativity_gamma: float):
    return back_end.em_kernel_cuda.multislice_propagate_cuda(
        wave_in, slices,
        pixel_size, dz, wave_length, relativity_gamma)


@requires_cuda_ext
def lens_propagate_cuda(wave_in: back_end.cp.ndarray, pixel_size: float,
                        wave_length: float, cs_mm: float, defocus: float, aperture: float):
    return back_end.em_kernel_cuda.lens_propagate_cuda(
        wave_in, pixel_size,
        wave_length, cs_mm, defocus, aperture)


def _make_mesh_grid_fourier_space(pixel_size: float, size: Tuple[int, int]):
    q_max = 0.5/pixel_size
    qx_range = np.linspace(-q_max, q_max, size[0])
    qy_range = np.linspace(-q_max, q_max, size[1])
    qx, qy = np.meshgrid(qx_range, qy_range, indexing="ij")
    return qx, qy

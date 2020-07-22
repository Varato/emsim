"""
The functions to build electron density / atomic potential for bio molecules
"""

import numpy as np

from . import atoms as atm
from . import elem


def fourier_potential_builder(atmv: atm.AtomVolume, projected: bool = True) -> np.ndarray:
    """
    builds atomic potential by convolution in Fourier space.

    Parameters
    ----------
    atmv: AtomVolume object
    projected: bool
        If true, then this function builds projected potential slices.
        If false, then it builds 3D potential.
        Default is True.

    Returns
    -------
    slices or potential: 3D numpy array

    Notes
    -----
        The slices are built along z direction, i.e. the last dimension is for different slices.
    """
    elem_nums, histograms = atmv.unique_elements, atmv.atom_histograms
    dx, dy, dz = atmv.voxel_size
    len_x, len_y, len_z = atmv.box_size

    if projected:
        if dx != dy:
            raise ValueError("pixels must be squares")

        slices = np.zeros(shape=(len_x, len_y, len_z), dtype=np.float64)
        form_fac = elem.scattering_factors2d(elem_nums, dx, size=(len_x, len_y))

        for s in range(len_z):
            if not np.any(histograms[..., s]):
                continue
            for i, _ in enumerate(elem_nums):
                # with this setup, the potential has dimension [E] * [L]
                slices[..., s] += np.fft.ifft2(np.fft.fft2(histograms[i, :, :, s]) * form_fac[i]).real
                # slices[..., s] += np.fft.irfft2(np.fft.rfft2(histograms[i, :, :, s]) * form_fac[i, :, :len_y//2+1],
                #                                 s=(len_x, len_y))

        slices = np.where(slices < 1e-7, 1e-7, slices)
        return slices
    else:
        if not dx == dy == dz:
            raise ValueError("voxels must be cubes")

        potential = np.zeros(shape=(len_x, len_y, len_z), dtype=np.float64)
        form_fac = elem.scattering_factors(elem_nums, dx, size=(len_x, len_y, len_z))
        for i, z in enumerate(elem_nums):
            potential += np.fft.irfftn(np.fft.rfftn(histograms[i]) * form_fac[i, :, :, :len_z//2+1],
                                       s=(len_x, len_y, len_z))

        potential = np.where(potential < 1e-7, 1e-7, potential)
        return potential

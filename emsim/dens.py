"""
The functions to build electron density / atomic potential for bio molecules
"""
from typing import Union, Optional, Tuple
import numpy as np

from . import atoms as atm
from . import elem


def build_potential_fourier(mol: atm.AtomList,
                            voxel_size: float,
                            box_size: Optional[Union[int, Tuple[int, int, int]]] = None) \
        -> np.ndarray:
    """

    Parameters
    ----------
    mol
    voxel_size
    box_size

    Returns
    -------
    """
    if box_size is None:
        box_size = (None, ) * 3
    else:
        if isinstance(box_size, int):
            box_size = (box_size, ) * 3
        elif isinstance(box_size, tuple) and len(box_size) == 3:
            pass
        else:
            raise ValueError("box_size is either a single int or a tuple of three ints")

    mol = atm.centralize(mol)
    atmv = atm.bin_atoms(mol, voxel_size, box_size)
    box_size_final = atmv.box_size
    len_z = box_size_final[2]
    elem_nums = atmv.unique_elements

    potential = np.zeros(shape=box_size_final, dtype=np.float)
    form_fac = elem.scattering_factors(elem_nums, voxel_size, size=box_size_final)
    for i, z in enumerate(elem_nums):
        potential += np.fft.irfftn(
            np.fft.rfftn(
                atmv.atom_histograms[i]) * form_fac[i, :, :, :len_z // 2 + 1], s=box_size_final)

    np.clip(potential, a_min=1e-7, a_max=None, out=potential)
    return potential


def build_slices_fourier(mol: atm.AtomList,
                         pixel_size: float,
                         thickness: float,
                         frame_size: Optional[Union[int, Tuple[int, int]]] = None,
                         n_slices: Optional[int] = None):
    """

    Parameters
    ----------
    mol
    pixel_size
    thickness
    frame_size
    n_slices

    Returns
    -------

    """
    dims = [None, None, None]
    if frame_size is not None:
        if isinstance(frame_size, int):
            dims[0] = frame_size
            dims[1] = frame_size
        elif isinstance(frame_size, tuple) and len(frame_size) == 2:
            dims[0] = frame_size[0]
            dims[1] = frame_size[1]

    if isinstance(n_slices, int):
        dims[2] = n_slices

    mol = atm.centralize(mol)
    atmv = atm.bin_atoms(mol, voxel_size=(pixel_size, pixel_size, thickness), box_size=(dims[0], dims[1], dims[2]))

    elem_nums = atmv.unique_elements
    len_x, len_y = atmv.box_size[:2]
    if isinstance(n_slices, int):
        assert atmv.box_size[-1] == n_slices
    n_slices = atmv.box_size[-1]

    slices = np.zeros(shape=atmv.box_size, dtype=np.float)
    form_fac = elem.scattering_factors2d(elem_nums, pixel_size, size=(len_x, len_y))

    for s in range(n_slices):
        if not np.any(atmv.atom_histograms[..., s]):
            continue
        for i, _ in enumerate(elem_nums):
            # with this setup, the potential has dimension [E] * [L]
            # slices[..., s] += np.fft.ifft2(np.fft.fft2(atmv.atom_histograms[i, :, :, s]) * form_fac[i]).real
            slices[..., s] += np.fft.irfft2(
                np.fft.rfft2(
                    atmv.atom_histograms[i, :, :, s]) * form_fac[i, :, :len_y // 2 + 1], s=(len_x, len_y))

    np.clip(slices, a_min=1e-7, a_max=None, out=slices)
    return slices


def potential_patching(atml: atm.AtomList, projected=True, radius=3.0):
    """
    builds atomic potential by adding potential patches.

    Parameters
    ----------
    atmv: AtomList
    projected: bool
        If true, then this function builds projected potential slices.
        If false, then it builds 3D potential.
        Default is True.

    radius: int or float
        maximum radius of a single atomic potential (as the convolution kernel).

    Returns
    -------
    slices or potential: 3D numpy array

    Notes:
        The slices are built along z direction, i.e. the last dimension is for different slices.
    """


    atom_nums, atom_idx = atm.index_atoms(atml, )
    elem_nums = np.unique(atom_nums)

    dx, dy, dz = atmv.vox_size
    len_x, len_y, len_z = atmv.side_len

    if projected:
        if dx != dy:
            raise ValueError("pixels must be squares")

        # pre-calculate the projected potential patches for each unique element
        vz = atom.projected_potential(elem_nums, dx, radius)
        vz, patch_len = _regularize_potential_patch2d(vz, len_x, len_y)

        slices = np.zeros(shape=(len_x, len_y, len_z), dtype=np.float64)
        r = patch_len//2
        for i, z in enumerate(atom_nums):
            ix, iy, iz = atom_idx[i, :]
            start_x, end_x = ix-r, ix+r
            start_y, end_y = iy-r, iy+r

            # take care of atoms whose indices are out of range
            patch_start_x, patch_end_x = max(0, -start_x), min(patch_len, patch_len + len_x-1-end_x)
            patch_start_y, patch_end_y = max(0, -start_y), min(patch_len, patch_len + len_y-1-end_y)
            sx, ex = max(0, start_x), min(len_x, end_x+1)
            sy, ey = max(0, start_y), min(len_y, end_y+1)

            slices[sx:ex, sy:ey, iz] += vz[z][patch_start_x:patch_end_x, patch_start_y:patch_end_y]

        if atmv.is_hydrated:
            vox_wat_num = pc.water_num_dens * dx*dy*dz
            water_pot = atom.projected_potential_water(dx, radius)
            water_cube = np.where(atmv.vacancies,
                                  np.random.poisson(vox_wat_num, size=(len_x, len_y, len_z)), 0).astype(np.int)
            water_slices = np.empty(shape=(len_x, len_y, len_z), dtype=np.float)
            for s in range(len_z):
                water_slices[..., s] = signal.convolve(water_cube[..., s], water_pot, mode="same", method="auto")
            slices += water_slices
        return slices
    else:
        if not dx == dy == dz:
            raise ValueError("voxels must be cubes")

        # pre-calculate the 3D potential patches for each unique element
        v = atom.potential(elem_nums, dx, radius)
        v, patch_len = _regularize_potential_patch3d(v, len_x, len_y, len_z)

        potential = np.zeros(shape=(len_x, len_y, len_z), dtype=np.float64)
        r = patch_len//2
        for i, z in enumerate(atom_nums):
            ix, iy, iz = atom_idx[i, :]
            start_x, end_x = ix-r, ix+r
            start_y, end_y = iy-r, iy+r
            start_z, end_z = iz-r, iz+r

            # take care of atoms whose indices are out of range
            patch_start_x, patch_end_x = max(0, -start_x), min(patch_len, patch_len + len_x-1-end_x)
            patch_start_y, patch_end_y = max(0, -start_y), min(patch_len, patch_len + len_y-1-end_y)
            patch_start_z, patch_end_z = max(0, -start_z), min(patch_len, patch_len + len_z-1-end_z)

            sx, ex = max(0, start_x), min(len_x, end_x+1)
            sy, ey = max(0, start_y), min(len_y, end_y+1)
            sz, ez = max(0, start_z), min(len_z, end_z+1)

            potential[sx:ex, sy:ey, sz:ez] += v[z][patch_start_x:patch_end_x,
                                                   patch_start_y:patch_end_y,
                                                   patch_start_z:patch_end_z]

        return potential


def _regularize_potential_patch2d(vz, len_x, len_y):
    # crop the projected potential patches if the patches are larger than the slices (should rarely happen)

    elem_nums = list(vz.keys())

    init_patch_len = vz[elem_nums[0]].shape[-1]
    patch_len = min(init_patch_len, len_x, len_y)
    patch_len = patch_len - 1 if patch_len % 2 == 0 else patch_len
    if patch_len < init_patch_len:
        diff = init_patch_len - patch_len  # must be an even number
        start = diff//2
        for z in vz.keys():
            vz[z] = vz[z][start:start+patch_len, start:start+patch_len]
    return vz, patch_len


def _regularize_potential_patch3d(v, len_x, len_y, len_z):
    # crop the potential patches if the patches are larger than the slices (should rarely happen)

    elem_nums = list(v.keys())
    init_patch_len = v[elem_nums[0]].shape[-1]
    patch_len = min(init_patch_len, len_x, len_y, len_z)
    patch_len = patch_len - 1 if patch_len % 2 == 0 else patch_len
    if patch_len < init_patch_len:
        diff = init_patch_len - patch_len  # must be an even number
        start = diff//2
        for z in v.keys():
            v[z] = v[z][start:start+patch_len, start:start+patch_len, start:start+patch_len]

    return v, patch_len

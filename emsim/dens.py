"""
The functions to build electron density / atomic potential for bio molecules
"""
from typing import Union, Optional, Tuple
import numpy as np

from . import atoms as atm
from . import elem


def build_potential_fourier(mol: atm.AtomList,
                            voxel_size: float,
                            box_size: Optional[Union[int, Tuple[int, int, int]]] = None,
                            add_water: bool = False) \
        -> np.ndarray:
    """

    Parameters
    ----------
    mol
    voxel_size
    box_size
    add_water

    Returns
    -------
    array
        the 3D atom potential of the molecule
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
    if add_water:
        atmv = atm.add_water_simple(atmv)

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
                         n_slices: Optional[int] = None,
                         add_water: bool = False):
    """
    builds projected potential slices for EM multi-slice imaging simulation

    Parameters
    ----------
    mol
    pixel_size
    thickness
    frame_size
    n_slices
    add_water

    Returns
    -------
    array
        projected potential slices

    Notes
    -----
    The third dimension indexes different slices. For example:
    `slices = build_slices_fourier(...)`, then `slices[..., i]` is the i-th slice.

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
    if add_water:
        atmv = atm.add_water_simple(atmv)

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


def build_slices_patchins(mol: atm.AtomList,
                          pixel_size: float,
                          thickness: float,
                          frame_size: Optional[Union[int, Tuple[int, int]]] = None,
                          n_slices: Optional[int] = None,
                          radius: float = 3.0):

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

    len_x, len_y, len_z = dims

    elem_nums, atom_idx = atm.index_atoms(
        mol, voxel_size=(pixel_size, pixel_size, thickness), box_size=(len_x, len_y, len_z))

    unique_elems = np.unique(elem_nums)

    vz = elem.projected_potentials(unique_elems, pixel_size, radius)
    vz, patch_len = _regularize_potential_patch2d(vz, len_x, len_y)

    slices = np.zeros(shape=(len_x, len_y, len_z), dtype=np.float64)
    r = patch_len // 2
    for i, z in enumerate(elem_nums):
        ix, iy, iz = atom_idx[i, :]
        start_x, end_x = ix - r, ix + r
        start_y, end_y = iy - r, iy + r

        # take care of atoms whose indices are out of range
        patch_start_x, patch_end_x = max(0, -start_x), min(patch_len, patch_len + len_x - 1 - end_x)
        patch_start_y, patch_end_y = max(0, -start_y), min(patch_len, patch_len + len_y - 1 - end_y)
        sx, ex = max(0, start_x), min(len_x, end_x + 1)
        sy, ey = max(0, start_y), min(len_y, end_y + 1)

        slices[sx:ex, sy:ey, iz] += vz[z][patch_start_x:patch_end_x, patch_start_y:patch_end_y]

    return slices


def _regularize_potential_patch2d(vz, len_x, len_y):
    """
    crops the projected potential patches if the patches are larger than the slices (should rarely happen)


    Parameters
    ----------
    vz
    len_x
    len_y

    Returns
    -------

    """

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

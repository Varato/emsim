from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
from functools import reduce

from .rotation import get_rotation_mattrices


class AtomList(object):
    def __init__(self, elements: np.ndarray, coordinates: np.ndarray):
        """
        Parameters
        ----------
        elements: array
            in shape (n_elems, ).
            specifies elements by their element number Z.
        coordinates: array
            in shape (n_elems, 3).
            specifies batched coordinates for the elements.
        """
        self.elements = elements
        self.coordinates = coordinates

    @property
    def r_min(self):
        return np.min(self.coordinates, axis=0)

    @property
    def r_max(self):
        return np.max(self.coordinates, axis=0)

    @property
    def space(self):
        return self.r_max - self.r_min


def group_atoms(atom_list: AtomList) -> Dict[int, np.ndarray]:
    """
    groups atom_list by element kinds.

    Parameters
    ----------
    atom_list: AtomList

    Returns
    -------
    dict. Keys are element numbers Z, values are all x,y,z coordinates of this kind.
    """
    atml = sort_atoms(atom_list)
    elems, elem_count = np.unique(atml.elements, return_counts=True)
    elem_dict = {}
    m = 0
    for i, z in enumerate(elems):
        n = elem_count[i]
        elem_dict[z] = atml.coordinates[m:m+n]
        m += n
    return elem_dict


def sort_atoms(atom_list: AtomList) -> AtomList:
    idx = np.argsort(atom_list.elements)
    sorted_elems = atom_list.elements[idx]
    sorted_coords = atom_list.coordinates[idx]
    return AtomList(elements=sorted_elems, coordinates=sorted_coords)


def translate(mol: AtomList, r: np.ndarray) -> AtomList:
    return AtomList(elements=mol.elements, coordinates=mol.coordinates + r)


def centralize(mol: AtomList) -> AtomList:
    return translate(mol, -mol.r_min-mol.space/2.0)


def rotate(mol: AtomList, quat: np.ndarray, set_center: bool = False) -> AtomList:
    rot = get_rotation_mattrices(quat)
    if set_center:
        mol = centralize(mol)
    rot_coords = np.matmul(rot, mol.coordinates[..., None]).squeeze()
    return AtomList(elements=mol.elements, coordinates=rot_coords)


def bin_atoms(mol: AtomList,
              voxel_size: Union[float, Tuple[float, float, float]],
              box_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    """
    puts atoms in bins (3D histogram). The process is applied to each kind of element separately.
    The bins are determined by voxel_size and molecule size / box_size.

    Parameters
    ----------
    mol: AtomList
    voxel_size: Union[float, Tuple[float, float, float]]
    box_size: Optional[Union[int, Tuple[int, int, int]]]

    Returns
    -------
    Dict[int, np.ndarray]
        keys specify elements by their Z number.
        values are the the binning volumes for each kind of element.

    Notes
    -----
    The (x, y, z) coordinates of the input molecule atoms should have origin at the geometric center of the molecule.
    Otherwise the binned molecule will not be placed at the center of the box.

    """
    bins = _find_bin_edges(voxel_size, mol.space, box_size)
    elem_dict = group_atoms(mol)

    volume = {}
    for z in elem_dict:
        xyz = elem_dict[z]
        h, _ = np.histogramdd(xyz, bins=bins)
        volume[z] = h.astype(np.int)

    return volume


def index_atoms(mol: AtomList,
                voxel_size: Union[float, Tuple[float, float, float]],
                box_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    """
    The inverted indexing process of the `bin_atoms`.
    In bin_atoms, we find what atoms are in a given bin. Here we find for each atom which bin it belongs to.
    The bin is specified by indices along the 3 dimensions.

    Parameters
    ----------
    mol: AtomList
    voxel_size: Union[float, Tuple[float, float, float]]
    box_size: Optional[Union[int, Tuple[int, int, int]]]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The elements Z value and their corresponding bin indices.

    """

    bins = _find_bin_edges(voxel_size, mol.space, box_size)
    num_bins = [len(bins[d]) - 1 for d in range(3)]
    xyz = mol.coordinates

    # the index is 1 at the first bin. hence -1.
    x_idx = np.digitize(xyz[:, 0], bins[0], right=False) - 1
    y_idx = np.digitize(xyz[:, 1], bins[1], right=False) - 1
    z_idx = np.digitize(xyz[:, 2], bins[2], right=False) - 1

    atom_idx = np.transpose(np.vstack((x_idx, y_idx, z_idx)))

    # remove atoms out of the box
    valid = reduce(np.logical_and, (np.min(atom_idx, axis=1) >= 0,
                                    atom_idx[:, 0] < num_bins[0],
                                    atom_idx[:, 1] < num_bins[1],
                                    atom_idx[:, 2] < num_bins[2]))
    if not np.any(valid):
        raise ValueError("No atoms in roi, or the thickness_nm is too small")

    return mol.elements[valid], atom_idx[valid]


def _find_bin_edges(voxel_size: Union[float, Tuple[float, float, float]],
                    real_size: Tuple[float, float, float],
                    box_size: Optional[Union[int, Tuple[int, int, int]]] = None) \
        -> Tuple[Any, ...]:

    """
    Find bins for histogramdd for atom (x, y, z) coordinates.
    This assumes the origin is at the geometric center of the molecule, so that the binned
    atoms are at the center of the box.

    Parameters
    ----------
    voxel_size: Union[float, Tuple[float, float, float]]
        this equals to the bin size.
    real_size: Tuple[float, float, float]
        the molecules geometric size
    box_size: Optional[Union[int, Tuple[int, int, int]]]
        target box size of the resulted atom volume.
        If not given, then the function uses the voxel_size and real_size to compute a minimum box_size.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]: the bins for the 3 dimensions.

    """

    if type(voxel_size) is float:
        voxel_size = (voxel_size, voxel_size, voxel_size)

    # regularise max_len
    if box_size is not None:
        if type(box_size) is int:
            box_size = (box_size, box_size, box_size)
    else:
        box_size = np.ceil(np.array(real_size) / voxel_size).astype(np.int)

    start = [
        (lambda n, dx: -dx * (n // 2 - 1) if n % 2 == 0 else -dx * (n // 2))(
            box_size[d] + 1, voxel_size[d])
        for d in range(3)]
    end = [
        (lambda n, dx: dx * (n // 2))(
            box_size[d] + 1, voxel_size[d])
        for d in range(3)]

    bins = tuple(np.arange(start[d], end[d] + voxel_size[d], voxel_size[d]) for d in range(3))
    return bins

from typing import Union, Optional, Tuple, Dict
import numpy as np

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
    return translate(mol, -mol.r_min-mol.space)


def rotate(mol: AtomList, quat: np.ndarray) -> AtomList:
    rot = get_rotation_mattrices(quat)
    rot_coords = np.matmul(rot, mol.coordinates[..., None]).squeeze()
    return AtomList(elements=mol.elements, coordinates=rot_coords)


def bin_atoms(mol: AtomList,
              voxel_size: Union[float, Tuple[float, float, float]],
              box_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    bins = _find_bin_edges(voxel_size, mol.space, box_size)
    mol = translate(mol, -mol.r_min)  # move origin to the corner
    elem_dict = group_atoms(mol)

    volume = {}
    for z in elem_dict:
        xyz = elem_dict[z]
        h, _ = np.histogramdd(xyz, bins=bins)
        volume[z] = h.astype(np.int)

    return volume


def _find_bin_edges(voxel_size: Union[float, Tuple[float, float, float]],
                    real_size: Tuple[float, float, float],
                    box_size: Optional[Union[int, Tuple[int, int, int]]] = None):

    if type(voxel_size) is float:
        voxel_size = (voxel_size, voxel_size, voxel_size)

    # regularise max_len
    if box_size is not None:
        if type(box_size) is int:
            box_size = (box_size, box_size, box_size)
    else:
        box_size = np.ceil(np.array(real_size) / voxel_size).astype(np.int)

    bins = tuple(np.arange(0, box_size[d] * voxel_size[d], voxel_size[d]) for d in range(3))

    return bins

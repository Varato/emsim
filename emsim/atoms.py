"""
This module provides functions that deal with atomic representations of molecules.

AtomList: Represents a list of atoms specified by their Z, associated with their corresponding (x, y, z) coordinates.
AtomVolume: Represents a list of unique elements specified by their Z, associated with all atoms of this kind binned in
    a 3D histogram according to their (x, y, z) coordiantes.
"""
from typing import Union, Optional, Tuple, List, Generator, Set
from functools import reduce
import math
import numpy as np
import bisect


from .utils.rot import get_rotation_mattrices
from .physics import water_num_dens

float_type = np.float32


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
        self.coordinates = coordinates.astype(float_type)

    @property
    def r_min(self):
        return self.coordinates.min(axis=0)

    @property
    def r_max(self):
        return self.coordinates.max(axis=0)

    @property
    def space(self):
        return self.r_max - self.r_min


def sort_elements_and_count(atom_list: AtomList, must_include_elems: Optional[List[int]] = ()):
    """
    sort the atom_list according to elements. and count the number of occurance of each elements.

    Parameters
    ----------
    atom_list: AtomList
    must_include_elems: List[int]
        If given, the returned unique elements must include the elements in the must_include_elems list.
        If an element in must_include_elems does not occur in atom_list, then 0 is added to corresponding
        position in the returned unique_elements_counts.

    Returns
    -------
    AtomList: the sorted AtomList
    np.ndarray: sorted unique elements of the returned AtomList
    np.ndarray: the corresponding number of occurance of each unique element
    """

    idx = np.argsort(atom_list.elements)
    elements = atom_list.elements[idx]
    coordinates = atom_list.coordinates[idx]

    # np.unique returns sorted unique values
    unique_elements, unique_elements_counts = np.unique(elements, return_counts=True)

    key_elements_sorted = sorted(list(must_include_elems))
    for key_z in key_elements_sorted:
        if key_z not in unique_elements:
            i = bisect.bisect_left(unique_elements, key_z)
            unique_elements = np.insert(unique_elements, i, key_z)
            unique_elements_counts = np.insert(unique_elements_counts, i, 0)

    return AtomList(elements=elements, coordinates=coordinates), unique_elements, unique_elements_counts


def sort_by_coordinates(atom_list: AtomList, axis=0):
    key_coord = atom_list.coordinates.T[axis]
    idx = np.argsort(key_coord)
    elements = atom_list.elements[idx]
    coordinates = atom_list.coordinates[idx]
    return AtomList(elements=elements, coordinates=coordinates)


class AtomVolume(object):
    def __init__(self,
                 unique_elements: List[int],
                 atom_histograms,
                 voxel_size: Union[float, Tuple[float, float, float]]):
        self.unique_elements = unique_elements  # (n_elems, )
        self.atom_histograms = atom_histograms  # (n_elems, *box_size)
        if type(voxel_size) is float:
            self.voxel_size = (voxel_size, voxel_size, voxel_size)
        else:
            self.voxel_size = voxel_size
        self.n_elems = len(unique_elements)
        self.box_size = self.atom_histograms.shape[-3:]

    @property
    def vacancies(self):
        vacs = np.where(self.atom_histograms == 0, True, False)
        return np.logical_and.reduce(vacs, axis=0)


def concatenate(atmls: List[AtomList]):
    elems_concat = np.concatenate([atml.elements for atml in atmls])
    coord_concat = np.concatenate([atml.coordinates for atml in atmls])
    return AtomList(elements=elems_concat, coordinates=coord_concat)


def translate(atom_list: AtomList, r) -> AtomList:
    """
    translates all atoms by a displacement vector.

    Parameters
    ----------
    atom_list: AtomList
        the input AtomList.
    r: ndarray
        the displacement vector.

    Returns
    -------
    AtomList
        the translated AtomList.
    """
    return AtomList(elements=atom_list.elements, coordinates=atom_list.coordinates + r)


def centralize(atom_list: AtomList) -> AtomList:
    """
    puts the origin of the system at the geometric center of the input AtomList.

    This is majorly used int `bin_atoms()` so that the resulted system is in the certer
    of the histogram box.

    Parameters
    ----------
    atom_list: AtomList
        the input AtomList.

    Returns
    -------
    AtomList
        the resulted AtomList.
    """
    return translate(atom_list, -atom_list.r_min-atom_list.space/2.0)


def rotate(atom_list: AtomList, quat: np.ndarray, set_center: bool = False) -> AtomList:
    """
    rotates the input AtomList by a SO3 operation specified by a quaternion.
    This function only supports a single rotation. For multiple rotations, use
    the generator `orientations_gen`.

    Parameters
    ----------
    atom_list: AtomList
    quat: np.ndarray
        one-single quaternion in shape (4, ).
    set_center: bool
        if set True, then the origin is first assigned to the geometrical center of the system
        before the rotation. Default False.
    Returns
    -------
    AtomList
        the rotated AtomList.
    """
    if set_center:
        atom_list = centralize(atom_list)
    rot = get_rotation_mattrices(quat)
    rot_coords = np.matmul(rot, atom_list.coordinates[..., None]).squeeze()
    atml = AtomList(elements=atom_list.elements, coordinates=rot_coords)
    return atml


def orientations_gen(atom_list: AtomList, quats: np.ndarray, set_center: bool = False) \
        -> Generator[AtomList, None, None]:
    """
    A generator that yield rotated systems specified by different AtomLists.

    Parameters
    ----------
    atom_list: AtomList
        the input AtomList
    quats: array
        multiple quaternions in a 2D array in shape (batch_size, 4)

    set_center: bool
        if set True, then the origin is first assigned to the geometrical center of the system
        before the rotation. Default False.

    Returns
    -------
    Generator
        Yields rotated AtomLists.

    """
    for i in range(quats.shape[0]):
        yield rotate(atom_list, quats[i], set_center)


def determine_box_size(space: Tuple[float, float, float],
                       voxel_size: Union[float, Tuple[float, float, float]],
                       box_size: Tuple[Union[None, int], Union[None, int], Union[None, int]]):
    space = [3.0 if space[d] < 1e-3 else space[d] for d in range(3)]
    if type(voxel_size) is float:
        voxel_size = (voxel_size, voxel_size, voxel_size)

    box = [int(math.ceil((length/d))) for length, d in zip(space, voxel_size)]
    box_final = tuple(box[d] if box_size[d] is None else box_size[d] for d in range(3))
    return box_final


def find_slices(mol: AtomList, thickness: float, n_slices: int = None, axis: int = 0) \
        -> Tuple[List[Tuple[AtomList, np.ndarray]], np.ndarray]:
    """
    groups atoms by which slices they belong to. The grouped atoms form seperate AtomLists

    Parameters
    ----------
    mol: AtomList
        the input atoms to be divided into slices
    thickness: float
        the thickness in Angstrom of each slice
    n_slices: int
        specifies how many slices to be grouped to.
    axis: int
        must be 0, 1, 2. Specifies along which axis the slices are divided

    Returns
    -------
    slices: List[Tuple[AtomList, np.ndarray]]
        Each slice in the list is defined as follows.
        AtomList: contains all atom coordinates in that slice and the corresponding element Z-values.
        unique_elements_counts: the number of occurance of the corresponding elements defined in.

    all_unique_elements: np.ndarray
        contains all unique elements in the system, in Z-value-sorted order.
        Notice that unique_elements_counts of each slice has the same length as all_unique_elements.
        If a slice does not contain an element in all_unique_elements, then the corresponding count is 0.
    """

    mol, all_unique_elements, all_unique_elements_counts = sort_elements_and_count(mol, must_include_elems=[1, 8])
    mol.sort_by_coordinates(axis)

    bin_edges = _find_bin_edges1(thickness, mol.space[axis], n_slices)
    n_bins = len(bin_edges) - 1
    key_coord = mol.coordinates.T[axis]

    # digitize returns 0 if the value is out of the left boundary
    # returns len(bin_edges) if the value is out of the right boundary
    slc_idx = np.digitize(key_coord, bin_edges, right=False)  # idx
    unique_slc_idx, unique_slc_cnt = np.unique(slc_idx, return_counts=True)

    slices = []
    m = 0
    for i, s in enumerate(unique_slc_idx):
        n = unique_slc_cnt[i]
        if not (s == 0 or s == n_bins):
            coord_in_slc = mol.coordinates[m:m+n]
            elems_in_slc = mol.elements[m:m+n]
            slc_atml = AtomList(elements=elems_in_slc, coordinates=coord_in_slc)
            slc_atml_sorted, _, uec = sort_elements_and_count(slc_atml, must_include_elems=all_unique_elements)
            slices.append((slc_atml_sorted, uec))
        m += n
    return slices, all_unique_elements


def index_atoms(mol: AtomList,
                voxel_size: Union[float, Tuple[float, float, float]],
                box_size: Tuple[Union[None, int], Union[None, int], Union[None, int]]):
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
        raise ValueError("No atoms in roi, or the thickness is too small")

    return mol.elements[valid], atom_idx[valid]


def add_water_simple(atmv: AtomVolume) -> AtomVolume:
    """
    Simply add oxygens and hydrogens in vacancies of the given AtomVolume.

    Parameters
    ----------
    atmv: AtomVolume
        the input AtomVolume

    Returns
    -------
    AtomVolume
        the resulted AtomVolume with oxygens and hydrogens in vacancy voxels

    """
    dx, dy, dz = atmv.voxel_size
    vox_wat_num = water_num_dens * dx*dy*dz  # average number of water molecules in a voxel

    oxygens = np.where(atmv.vacancies, np.random.poisson(vox_wat_num, atmv.box_size), 0).astype(np.int)
    hydrogens = np.where(atmv.vacancies, np.random.poisson(vox_wat_num * 2, atmv.box_size), 0).astype(np.int)

    unique_elements = atmv.unique_elements
    atom_histograms = atmv.atom_histograms
    for z, hist in [(1, hydrogens), (8, oxygens)]:
        if z in unique_elements:
            idx = unique_elements.index(z)
            atom_histograms[idx] += hist
        else:
            unique_elements.append(z)
            atom_histograms = np.append(atom_histograms, hist[None, ...], axis=0)

    return AtomVolume(unique_elements, atom_histograms, atmv.voxel_size)


def _find_bin_edges1(d: float, length: float, n_bins: Optional[int] = None) -> np.ndarray:
    """
    Given a bin size d and a space length L, this function finds the bin edges
    that by default roughly cover the interval [-0.5L, 0.5L]. By given n_bins, the covering
    range canbe be longer or shorter.

    Parameters
    ----------
    d: float
        bin size
    length: float
        space length
    n_bins: Optional[int]

    Returns
    -------
    np.ndarray
        the bin edges
    """
    if n_bins is None:
        n_bins = int(math.ceil(length / d))

    n_edges = n_bins + 1

    start = -d * (n_edges // 2)
    end = d * (n_edges // 2)
    return np.arange(start, end + d, d)[:n_edges]


def _find_bin_edges(voxel_size: Union[float, Tuple[float, float, float]],
                    lengths: Tuple[float, float, float],
                    box_size: Tuple[Union[None, int], Union[None, int], Union[None, int]]):

    if type(voxel_size) is float:
        voxel_size = (voxel_size, voxel_size, voxel_size)

    return tuple(_find_bin_edges1(voxel_size[i], lengths[i], box_size[i]) for i in range(3))

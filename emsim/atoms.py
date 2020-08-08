"""
This module provides functions that deal with atomic representations of molecules.

AtomList: Represents a list of atoms specified by their Z, associated with their corresponding (x, y, z) coordinates.
AtomVolume: Represents a list of unique elements specified by their Z, associated with all atoms of this kind binned in
    a 3D histogram according to their (x, y, z) coordiantes.
"""
from typing import Union, Optional, Tuple, List, Generator
from functools import reduce
import math
import numpy as np


from .utils.rot import get_rotation_mattrices
from .physics import water_num_dens

float_type = np.float32


class AtomList(object):
    def __init__(self, elements, coordinates):
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

        self._sorted = False
        self._unique_elements = None
        self._unique_elements_count = None

    @property
    def r_min(self):
        return self.coordinates.min(axis=0)

    @property
    def r_max(self):
        return self.coordinates.max(axis=0)

    @property
    def space(self):
        return self.r_max - self.r_min

    @property
    def unique_elements(self):
        self.sort()
        return self._unique_elements

    @property
    def unique_elements_count(self):
        self.sort()
        return self._unique_elements_count

    def sort(self):
        if not self._sorted:
            idx = np.argsort(self.elements)
            self.elements = self.elements[idx]
            self.coordinates = self.coordinates[idx]
            self._unique_elements, self._unique_elements_count = np.unique(self.elements, return_counts=True)
            self._sorted = True
        return self

    def translate(self, r):
        atml = AtomList(elements=self.elements, coordinates=self.coordinates + r)
        atml._sorted = self._sorted
        return atml

    def rotate(self, quat):
        rot = get_rotation_mattrices(quat)
        rot_coords = np.matmul(rot, self.coordinates[..., None]).squeeze()
        atml = AtomList(elements=self.elements, coordinates=rot_coords)
        atml._sorted = self._sorted
        return atml


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


def sort_atoms(atom_list: AtomList) -> AtomList:
    """
    sorts and AtomList according to element numbers.

    Parameters
    ----------
    atom_list: AtomList
        the input AtomList.

    Returns
    -------
    AtomList
        the sorted AtomList.

    """
    return atom_list.sort()
    

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
    return atom_list.translate(r)


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
    return atom_list.rotate(quat)


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
    for quat in quats:
        yield rotate(atom_list, quat, set_center)


def bin_atoms(mol: AtomList,
              voxel_size: Union[float, Tuple[float, float, float]],
              box_size: Tuple[Union[None, int], Union[None, int], Union[None, int]]) \
        -> AtomVolume:
    """
    puts atoms in bins (3D histogram). The process is applied to each kind of element separately.
    The bins are determined by voxel_size and molecule size / box_size.

    Parameters
    ----------
    mol: AtomList
    voxel_size: Union[float, Tuple[float, float, float]]
    box_size: Tuple[Union[None, int], Union[None, int], Union[None, int]])
        specifies the 3 dimensions of the resulted histogram box.
        Whenever a dimension is given as None, this dimension is selected to be
        a minimum value that just covers the system.

    Returns
    -------
    AtomVolume
        keys specify elements by their Z number.
        values are the the binning volumes for each kind of element.

    Notes
    -----
    The (x, y, z) coordinates of the input molecule atoms should have origin at the geometric center of the molecule.
    Otherwise the binned molecule will not be placed at the center of the box.

    """
    space = np.where(mol.space < 1e-3, 3.0, mol.space)
    bins = _find_bin_edges(voxel_size, space, box_size)
    num_bins = [len(bins[d]) - 1 for d in range(3)]

    z_values = mol.unique_elements
    count = mol.unique_elements_count
    n_elems = len(z_values)

    volumes = np.empty(shape=(n_elems, *num_bins), dtype=np.int)

    m = 0
    for i in range(n_elems):
        n = count[i]
        volumes[i, ...], _ = np.histogramdd(mol.coordinates[m:m+n], bins=bins)
        m += n

    return AtomVolume(unique_elements=z_values, atom_histograms=volumes, voxel_size=voxel_size)


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
        raise ValueError("No atoms in roi, or the thickness_nm is too small")

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

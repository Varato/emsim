"""
build bio-molecule potential / scattering factor
"""
from typing import Union, Optional, Tuple, List
import numpy as np

from . import utils
from .rotation import get_rotation_mattrices
from .atom_list import AtomList


def rotate_generator(mol: AtomList):
    """

    Parameters
    ----------
    mol: namedtuple (elements, coordinates)
        a biomolecule specified by its elements and coordinates.

        This variable becomes a closure to the inner function rotate.

    Returns
    -------
    A callable which takes quaternions as input to produce copies of the molecule at different orientations.
    """
    def _rotate(quat: np.ndarray):
        """

        Parameters
        ----------
        quat: array
            batched quaternions in shape (n_rot, 4)

        Returns
        -------
        Generator
            that generates rotated copies of the molecule.
        """
        rot = get_rotation_mattrices(quat)  # (n_rot, 3, 3)
        n_rot = rot.shape[0]
        for i in range(n_rot):
            # (3, 3) @ (n_elems, 3, 1) -> (n_elems, 3, 1)
            rot_coords = np.matmul(rot[i], mol.coordinates[..., None]).squeeze()  # (n_elems, 3)
            yield AtomList(elements=mol.elements, coordinates=rot_coords)
    return _rotate




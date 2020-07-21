"""
build bio-molecule potential / scattering factor
"""

import numpy as np
from . import utils
from .rotation import get_rotation_mattrices
from .atom_list import AtomList


def rotate_molecule(mol: AtomList):
    """

    Parameters
    ----------
    mol: namedtuple (elements, coordinates)
        a biomolecule specified by its elements and coordinates.

        This variable becomes closure to the inner function rotate.

    Returns
    -------
    A callable which takes quaternions as input to produce copies of the molecule at different orientations.
    """

    def rotate(quat: np.ndarray):
        """

        Parameters
        ----------
        quat: array
            batched quaternions in shape (batch_size, 4)

        Returns
        -------
        AtomList: represents rotated recplica(s) of the molecule.
            The `elements` array remains the same, i.e. in shape (n_elems, ),
            while the `coordiantes` array becomes in shape (n_replicas, n_opers, n_elems, 3).

            The correspondence between `elements[i]` and `coordinates[:, :, i, :]` is interpreted as broadcasting.
        """
        batch_size = quat.shape[0]
        rot = get_rotation_mattrices(quat)  # (batch_size, 3, 3)
        # (batch_size, 1, 1, 3, 3) @ (n_ops, n_elems, 3, 1) -> (batch_size, n_ops, n_elems, 3, 1)
        rot_coords = np.matmul(rot[:, None, None, :, :], mol.coordinates[..., None]).squeeze()  # (batch_size, n_ops, n_elems, 3)
        return AtomList(elements=mol.elements, coordinates=rot_coords)

    return rotate


def bin_atoms(mol: AtomList):
    pass


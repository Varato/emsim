"""
This module contains a collection of utility functions for 3D rotations.
"""
import numpy as np
import math


def random_uniform_quaternion() -> np.ndarray:
    """
    Returns a random uniform quaternion as defined by

    h  = (h1,h2,h3,h4)
    h1 = sqrt(1-u1) sin(2 pi u2)
    h2 = sqrt(1-u1) cos(2 pi u2)
    h3 = sqrt(u1) sin(2 pi u3)
    h4 = sqrt(u1) cos(2 pi u3)

    u1,u2,u3 are random floats in [0,1]

    Returns
    -------
    ndarray of floats
         1x4 quaternion array

    Reference
    ---------
    K. Shoemake.
    Uniform random rotations.
    In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.
    """
    u1, u2, u3 = np.random.random(3)
    h1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    h2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    h3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    h4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.hstack([h1, h2, h3, h4])


def random_uniform_quaternions(n: int) -> np.ndarray:
    """
    Returns an array of random uniform quaternions

    Parameters
    ----------
    n : int
        Number of quaternions

    Returns
    -------
    ndarray of floats
        (n, 4) array of quaternions
    """
    return np.array([random_uniform_quaternion() for _ in range(n)])


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis:
    theta:
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_quaternion(n: np.ndarray, theta: float) -> np.ndarray:
    """
    converts a n-theta representation of rotation operator to a corresponding quaternion.

    Parameters
    ---------
    n: array-like,
        the rotation axis
    theta: float
        angle as rad
    Returns
    -------
        1D array in shape (4, )
    """

    if not isinstance(n, np.ndarray):
        if isinstance(n, (list, tuple)):
            n = np.array(n)
        else:
            raise ValueError("rotation axis n must be an array-like object.")
    if n.shape != (3,):
        raise ValueError("rotation axis n must be a 3D vector.")

    n = n / np.linalg.norm(n)
    quaternion = np.zeros(4, dtype=np.float64)
    quaternion[0] = np.cos(theta / 2)
    quaternion[1:] = np.sin(theta / 2) * n
    return quaternion


def get_rotation_mattrices(quat: np.ndarray) -> np.ndarray:
    """
    converts a quaternion into 3x3 rotation matrix.

    Parameters
    ---------
    quat: array
        batched quaternions in shape (batch_size, 4)

    Returns
    -------
    (batch_size, 3, 3) array
    """
    if quat.ndim != 2:
        raise ValueError("quat must be in shape (batch_size, 4)")

    n_rot = quat.shape[0]

    rot = np.empty(shape=(n_rot, 3, 3), dtype=np.float64)
    q0 = quat[:, 0]
    q1 = quat[:, 1]
    q2 = quat[:, 2]
    q3 = quat[:, 3]

    q01 = q0 * q1
    q02 = q0 * q2
    q03 = q0 * q3
    q11 = q1 * q1
    q12 = q1 * q2
    q13 = q1 * q3
    q22 = q2 * q2
    q23 = q2 * q3
    q33 = q3 * q3

    rot[:, 0, 0] = (1. - 2. * (q22 + q33))
    rot[:, 0, 1] = 2. * (q12 + q03)
    rot[:, 0, 2] = 2. * (q13 - q02)
    rot[:, 1, 0] = 2. * (q12 - q03)
    rot[:, 1, 1] = (1. - 2. * (q11 + q33))
    rot[:, 1, 2] = 2. * (q01 + q23)
    rot[:, 2, 0] = 2. * (q02 + q13)
    rot[:, 2, 1] = 2. * (q23 - q01)
    rot[:, 2, 2] = (1. - 2. * (q11 + q22))

    return rot


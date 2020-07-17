import numpy as np
from pathlib2 import Path
from collections import namedtuple
from typing import List


def atom_params(atom_numbers: List[int]) -> np.ndarray:
    """
    looks up potential parameters for queries by atom numbers.

    Parameters
    ----------
    atom_numbers: a list of Zs

    Returns
    -------
    ndarray with shape [num_queries, 13]. 13 = [chisqr, a1-3, b1-3, c1-3, d1-3]
    """

    if not all([1<=z<=103 for z in atom_numbers]):
        raise ValueError("z must be a interger in range [1, 92]")
    return np.array([_atom_params[z-1] for z in atom_numbers])

def atom_potentials(atom_numbers: List[int]) -> np.ndarray:
    pass


def _read_atom_parameters(asset_path: Path=None):
    """
    parses atom_params and return as numpy array.
    There are 12 parameters for each atom according to Appendix C.4 in Kirkland's book, 
    and they are listed in the following order:

    Z, chisqr
    a1, b1, a2, b2,
    a3, b3, c1, d1,
    c2, d2, c3, d3

    Parameters
    ----------
    asset_path: Path object

    Returns
    -------
    atom_pot_pm: ndarray
        The 12 parameters for each atom from H to U.
    """
    if asset_path is None:
        asset_path = Path(__file__).parent/"assets"

    with open(asset_path/"atom_params.txt", 'r') as f:
        raw_data = f.readlines()

    params = np.empty([103, 12+1])
    for Z in range(103):
        line1 = raw_data[Z*4]
        line2 = raw_data[Z*4+1]
        line3 = raw_data[Z*4+2]
        line4 = raw_data[Z*4+3]
        chiq = float(line1.split("=")[2].strip())
        abcd = [float(x) for x in (line2 + line3 + line4).strip().split()]
        params[Z, 0] = chiq
        params[Z, 1:4  ] = abcd[0:5:2]
        params[Z, 4:7  ] = abcd[1:6:2]
        params[Z, 7:10 ] = abcd[6:11:2]
        params[Z, 10:13] = abcd[7:12:2]

    return params

def _read_atom_mass(asset_path: Path=None):
    if asset_path is None:
        asset_path = Path(__file__).parent/"assets"

    with open(asset_path/"atom_mass.txt", 'r') as f:
        return np.array([float(line.split()[1]) for line in f])


_atom_params = _read_atom_parameters()

_element_symbol = [
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg',
    'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U']


if __name__ == "__main__":
    print(atom_params([1,2]))
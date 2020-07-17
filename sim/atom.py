import numpy as np
import math
from pathlib2 import Path
from typing import List, Tuple, Union


water_num_dens = 0.031    # number of molecules per A3
e = 14.39964              # electron charge in Volts * Angstrom
a0 = 0.529177             # Bohr radius in Angstrom
hbar_c = 1975.9086        # hbar*c in eV * Angstrom
hc = 12415                # h*c in eV * Angstrom
m0c2 = 510.9989461        # electron mass in keV

dtype = np.float32


def atom_params(atom_numbers: List[int]) -> np.ndarray:
    """
    looks up potential parameters for queries by element numbers.

    Parameters
    ----------
    atom_numbers: a list of Zs

    Returns
    -------
    ndarray with shape [num_queries, 13]. 13 = [chisqr, a1-3, b1-3, c1-3, d1-3]
    """

    if not all([1 <= z <= 103 for z in atom_numbers]):
        raise ValueError("z must be a interger in range [1, 92]")
    return np.array([_atom_params[z-1] for z in atom_numbers], dtype=dtype)


def atom_potentials(atom_numbers: List[int], voxel_size:float, radius=3.0) -> np.ndarray:

    # construct 3D meshgrid for the potential
    r0, r1, n = _potential_rspace_params(radius, voxel_size)
    xyz_range = np.linspace(r0, r1, n)
    xx, yy, zz = np.meshgrid(xyz_range, xyz_range, xyz_range, indexing="ij")
    r = np.sqrt(xx*xx + yy*yy + zz*zz)

    # find the singular point (the one that is nearest to the origin)
    singular_point_idx, displacement = _find_singular_point(r0, voxel_size, 0)
    # if the displacement from the origin is maller than half a voxel, consider it's singular. Cut it off.
    if displacement < 0.5:
        r[singular_point_idx, singular_point_idx, singular_point_idx] = 0.5 * voxel_size

    r2 = r**2

    c1 = 2 * np.pi**2 * a0 * e
    c2 = 2 * pow(np.pi, 5/2) * a0 * e

    n_atoms = len(atom_numbers)
    pms = atom_params(atom_numbers)
    v = np.empty(shape=(n_atoms, *r2.shape), dtype=dtype)
    for k in range(n_atoms):
        pm = pms[k]  # the 13 parameters
        a = pm[1:4]
        b = pm[4:7]
        c = pm[7:10]
        d = pm[10:]

        s1 = c1 * sum([a[i]/r * np.exp(-2*np.pi*r*np.sqrt(b[i])) for i in range(3)])
        s2 = c2 * sum([c[i]*pow(d[i], -3/2) * np.exp(-(np.pi**2) * r2/d[i]) for i in range(3)])

        v[k] = s1 + s2
    return v


def atom_scattering_factors(atom_numbers: List[int], voxel_size: float, size: Union[int, Tuple[int, int, int]]):
    # The following two functions are for fourier space convolution
    # The kernel size are designed to be given by caller, so that it's the same as
    # the convolved array.

    """
    pre-calculates 3D atomic scattering factors for 3D potential builder.
    This computation is based on equation (5.17) in Kirkland.

    Parameters
    ----------
    atom_numbers: list, atom numbers.
    voxel_size: float
    len_x: int
    len_y: int
    len_z: int

    Returns
    -------
    numpy array
        3D form factor(s) with the first dimension corresponding to atom_numbers
    """
    if type(size) is int:
        size = (size, size, size)

    fx_range = np.fft.fftshift(np.fft.fftfreq(size[0], voxel_size))
    fy_range = np.fft.fftshift(np.fft.fftfreq(size[1], voxel_size))
    fz_range = np.fft.fftshift(np.fft.fftfreq(size[2], voxel_size))

    fx, fy, fz = np.meshgrid(fx_range, fy_range, fz_range, indexing="ij")
    f2 = fx*fx + fy*fy + fz*fz
    mask = f2 < 16  # use frequency up to 4 A^-1

    factor = 2 * np.pi * e * a0 / voxel_size**3

    n_atoms = len(atom_numbers)
    pms = atom_params(atom_numbers)
    scat_fac = np.empty(shape=(n_atoms, size[0], size[1], size[2]), dtype=dtype)
    for k in range(n_atoms):
        pm = pms[k]  # the 13 parameters
        a = pm[1:4]
        b = pm[4:7]
        c = pm[7:10]
        d = pm[10:]

        scat_fac[k] = factor * np.sum([a[i] / (f2 + b[i]) + c[i] * np.exp(-d[i] * f2) for i in range(3)], axis=0)
        scat_fac[k] = np.where(mask, scat_fac[k], 0)
        scat_fac[k] = np.fft.ifftshift(scat_fac[k])

    return scat_fac


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

    params = np.empty([103, 12+1], dtype=dtype)
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


def _potential_rspace_params(r: float, dx: float) -> Tuple[float, float, int]:
    """
    This function slves the following problem.

    We want to sample atom potential (a 3D scalar field) up to a radius `r` at a certain sampling rate `dx`,
    while we want the sampling voxel box to have odd number of dimensions so that it can be properly place it
    in to a molecule's potential sampled space.

    If the number of samples from -r to r at dx is odd, then keep as it is.
    If the number of samples from -r to r at dx is even, 
        add one more sample by extending -r to -r-dx/2 and r to r+dx/2.

    Parameters
    ----------
    r: float
        the maximum radius to the potential center to sample within
    dx: float
        the sampling rate (voxel size)

    Returns
    -------
    start: float
        the coordinate of the starting point
    end: float
        the end point
    num: int
        the number of samples, which is designed to be odd
    """

    start = -r
    end = r
    num = math.floor((end - start)/dx) + 1
    if num % 2 == 0:
        start -= dx/2
        end += dx/2
        num += 1
    return start, end, num


def _find_singular_point(start: float, dx: float, x: float = 0):

    """
    Suppose `x` is singular point, this function finds the voxel that is nearest to origin, 
    and its displacement measured in framction of voxel size.
    
    Parameters
    ----------
    start: float
        the starting coordinate of the sampling linspace
    dx: float
        the sampling rate
    x: float
        the coordinate of the singular point

    Returns
    -------
    idx: int
        the index of the voxel that is nearest to the origin
    displacement: float
        the difference between the pixel that is closest to origin in fraction of voxel size.

    """

    # find the index that is nearest to origin, and its difference from real origin
    idx = int(round((x-start)/dx))
    displacement = abs((x-start)/dx - idx)

    return idx, displacement


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
    import matplotlib.pyplot as plt
    v = atom_potentials([10], voxel_size=1.0)
    print(v.shape)
    plt.imshow(v[0].sum(0))
    plt.show()
"""
A utils module for reading PDB files.
See http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html for detailed specification of PDB files.
Here we focus on ATOM, HETATM entry for atoms, 
and REMARK 350 for transformats that are needed to generate the biomolecule
"""
import os
import numpy as np
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.request import urlretrieve
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from typing import List

# Some of the molecule fucntions should probably go back to the utils module
from .. import atom


def fetch_pdb_file(pdbcode: str, output='./', force: bool = False, assembly: bool = False) -> str:
    """
    Download a PDB file and save it in a given location.

    Parameters
    ----------
    pdbcode : str
        A valid PDB code
    output : str, optional
        The destination for the PDB file to be saved
    force : bool, optional
        Download PDB file even if it already exists
    assembly : bool, optional
        Download biological assembly

    Returns
    -------
    str
        Path to the saved PDB file 
    """
    url = "https://files.rcsb.org/download/{code}.pdb{assembly}.gz".format(code=pdbcode, assembly="1" if assembly else "")
    filename = Path(output) / "{}.pdb".format(pdbcode)  #"".join([output,'/',pdbcode, '.pdb'])
    if (os.path.isfile(filename)) and not force:
        return filename
    try:
        response = urlopen(url)
    except HTTPError:
        raise IOError("Error 404: {url} not found".format(url=url))
    compressed = BytesIO()
    compressed.write(response.read())
    compressed.seek(0)
    decompressed = GzipFile(fileobj=compressed, mode='rb')
    with open(filename, "wb") as f:
        f.write(decompressed.read())
    return filename


def read_atoms_and_coordinates(pdb_file, residual=False, multimodel=False, assemble=True):
    """
    Parse a PDB file and return atom labes and coordinates.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    residual : bool, optional
        Counts residual atoms, default is False
    multimodel : bool, optional
        Parse multiple models (sometimes biological assemblies are saved as models), default is False
    assemble : bool, optional
        Apply symmetry operations and return biological assembly, default is True

    Returns
    -------
    elements : ndarray
        Atom elements listed in the PDB file, specified by atom number 1-92
    coords : ndarray
        Atom coordinates (x, y, z) listed in the PDB file
    rescount : int
        Return count of residual atoms if residual=True
    """

    legal_atoms = atom.symbols() # element #1-92 (H - U)
    elements, coords = [],[]
    rescount = 0
    with open(pdb_file) as f:
        for line in f:
            if (line[:6] == 'ENDMDL') and not multimodel:
                break
            if line.startswith("ATOM") or line.startswith("HETATM") or (line.startswith('ANISOU')):
                atom_label = line[76:78].lstrip().upper()
                (occ, tag) = (float(line[56:60]), line[16])
                use_atom = (occ > 0.5) | ((occ == 0.5) & (tag.upper() == "A"))
                if use_atom and (atom_label in legal_atoms):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elements.append(atom.number(atom_label))
                    coords.append([x,y,z])
                else:
                    rescount += 1

    coords = np.array(coords)
    if assemble:
        symmetries  = read_symmetry(pdb_file)
        elements, coords = apply_symmetry(elements, coords, symmetries)
    out = (elements, coords)
    if residual: out += (rescount,)
    return out


def read_symmetry(pdb_file: str) -> np.ndarray:
    """
    gets symmetry operators from a pdb file.

    Returns
    numpy array
        symmetry operators stored in 3 by 4 arrays. The first 3 columns are rotation, the last is translation.
        It should at least contains the identity.

    Notes
    -----
        The first symmetry operation is identity in normal pdb files, followed by non-trivial symmetries.
    """
    symm_list = []
    with open(pdb_file) as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            if line[13:18] == "BIOMT":
                symm_list.append(list(map(float, (line[24:33], line[34:43], line[44:53], line[58:68]))))

    if len(symm_list) == 0:
        # at least add the identity
        symms = np.hstack((np.identity(3), [[0], [0], [0]]))[np.newaxis, :]
    else:
        symms = np.array(symm_list).reshape((-1, 3, 4))
    return symms


def apply_symmetry(atom_numbers: List[int] , atom_xyz: np.ndarray, symmetries: np.ndarray):
    """
    applies symmetry operators on atoms to generate whole structure of pdb molecules.
    Here the symmetries must be in shape (num_symms, 3, 4), meaning that the rotation matrices and
    translation vectors are put together.

    Parameters
    ---------
    atom_numbers: List
        specifies a list of atoms by their Z
    atom_xyz: numpy array
        atoms' xyz coordinates
    symmetries: numpy array
        symmetry operators (integrated rotation and translation) in shape (num_symms, 3, 4)

    Returns
    -------
        new atomic symbols and xyz coordinates after symm_oper applied
    """

    num_symms = symmetries.shape[0]
    num_atoms = len(atom_numbers)

    # append the coordinates with ones, so as to perform rotation and translation simultaneously.
    tmp_atom_xyz = np.append(atom_xyz, np.ones((num_atoms, 1)), axis=1)
    all_atom_xyz = np.transpose(symmetries @ tmp_atom_xyz.T, (0, 2, 1))

    # the atom_symbols replicated according to number of symmetries.
    all_atom_numbers = atom_numbers * num_symms
    return all_atom_numbers, all_atom_xyz.reshape((num_symms * num_atoms, 3))

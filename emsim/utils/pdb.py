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
from typing import Union, Optional, List
from collections import namedtuple

from Bio.PDB import PDBParser, PDBList, MMCIFParser

from .. import elem


AtomList = namedtuple("AtomList", ['elements', 'coordinates'])

# Most amino acids are composed only by C, H, O, N.
# Exceptions: CYS, MET have S. SEC has Se
STANDART_AMINO_ACIDS22 = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CSC', 'LYS', 'MET', 'PHE', 'PRO', 'PYL', 'GLN',
    'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'SEC', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AMIBIGUOUS_AMINO_ACIDS = ['GLX', 'ASX', 'UNK']  # GLX = GLU or GLN, ASX = ASN or ASP, UNK = UNKNOWN
STANDART_NUCLEOTIDES11 = ['A', 'C', 'G', 'I', 'U', 'DA', 'DC', 'DG', 'DI', 'DT', 'DU', 'N']  # N = UNKNOWN

# Te standard residues are recored as ATOM record in PDB or PDBx/mmCIF
STANDART_RESIDUES = STANDART_AMINO_ACIDS22 + AMIBIGUOUS_AMINO_ACIDS + STANDART_NUCLEOTIDES11


def fetch_pdb_file(pdb_code: str, pdir: Union[Path, str] = Path('.'), overwrite: bool = False) -> str:
    """

    Parameters
    ----------
    pdb_code: str
        the PDB code to download
    pdir: the location to store the dowloaded file
    overwrite: bool
        specifies whether to overwrite existing files

    Returns
    -------
    filename: str

    """
    pdbl = PDBList(server="ftp://ftp.wwpdb.org")
    return pdbl.retrieve_pdb_file(pdb_code=pdb_code, pdir=pdir, file_format='mmCif', overwrite=overwrite)


def fetch_all_pdb_file(pdir):
    pass


def read_atoms(pdb_file: Union[str, Path], sort: bool = True, identifier: Optional[str] = None) -> AtomList:
    """
    read atoms from a PDB, or PDBx/mmCif file

    Parameters
    ----------
    pdb_file: Union[str, Path]
    sort: bool
        sort the read atoms by their element numbers. Their coordinates are correspondingly ordered.
    identifier: Optional[str]

    Returns
    -------
    AtomList: a namedtuple with two attributes:
        elements: the atoms specified by their element number
        coordinates: the corresponding coordinates
    """
    parser, code, _ = make_parser(pdb_file)

    if identifier is None:
        identifier = code

    structure = parser.get_structure(identifier, pdb_file)

    elems = []
    coords = []
    # for model in structure:
    #     for chain in model:
    #         for residue in chain:
    #             hetflag = residue.get_id()[0]
    #             resname = residue.get_resname()
    #             # for non standart residue
    #             if hetflag.strip():
    #                 for a in residue:
    #                     atom_name = a.get_name()
    #                     elem_name = ''.join([x for x in atom_name if x.isalpha()])
    #                     elems.append(elem.number(elem_name))
    #                     coords.append(a.get_coord())
    #             # for standard residule
    #             else:
    #                 if resname == 'SEC':  # SEC contains element Se
    #                     for a in residue:
    #                         atom_name = a.get_name()
    #                         elem_name = 'SE' if atom_name.startswith('S') else atom_name[0]
    #                         elems.append(elem.number(elem_name))
    #                         coords.append(a.get_coord())
    #                 else:  # others just contain C H O N P S
    #                     elems += [elem.number(a.get_name()[0]) for a in residue]
    #                     coords += [a.get_coord() for a in residue]

    for a in structure.get_atoms():
        z = elem.number(a.element)
        if z is None:
            continue
        elems.append(elem.number(a.element))
        coords.append(a.get_coord())

    atom_list = AtomList(elements=np.array(elems, dtype=np.int), coordinates=np.array(coords, dtype=np.float))
    if sort:
        return sort_atoms(atom_list)
    return atom_list


def sort_atoms(atom_list: AtomList) -> AtomList:
    idx = np.argsort(atom_list.elements)
    sorted_elems = atom_list.elements[idx]
    sorted_coords = atom_list.coordinates[idx]
    return AtomList(elements=sorted_elems, coordinates=sorted_coords)


def make_parser(pdb_file: Union[str, Path]):
    pdb_file = Path(pdb_file)
    if pdb_file.stem.startswith('pdb') and pdb_file.suffix == '.ent':
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
        code = pdb_file.stem[3:]
        file_format = 'pdb'
    elif pdb_file.suffix == '.cif':
        parser = MMCIFParser(QUIET=True)
        code = pdb_file.stem
        file_format = 'mmcif'
    else:
        raise ValueError("parser only supports PDB, PDBx/mmCif format")
    return parser, code, file_format


def read_symmetries(pdb_file: Union[str, Path]):
    parser, _, _ = make_parser(pdb_file)
    # parser.get_structure(pdb_file)
    # TODO




# def fetch_pdb_file(pdb_code: str, output='./', force: bool = False, assembly: bool = False) -> str:
#     """
#     Download a PDB file and save it in a given location.
#
#     Parameters
#     ----------
#     pdb_code : str
#         A valid PDB code
#     output : str, optional
#         The destination for the PDB file to be saved
#     force : bool, optional
#         Download PDB file even if it already exists
#     assembly : bool, optional
#         Download biological assembly
#
#     Returns
#     -------
#     str
#         Path to the saved PDB file
#     """
#     url = "https://files.rcsb.org/download/{code}.pdb{assembly}.gz".format(code=pdb_code, assembly="1" if assembly else "")
#     filename = Path(output) / "{}.pdb".format(pdb_code)  # "".join([output,'/',pdbcode, '.pdb'])
#     if (os.path.isfile(filename)) and not force:
#         return filename
#     try:
#         response = urlopen(url)
#     except HTTPError:
#         raise IOError("Error 404: {url} not found".format(url=url))
#     compressed = BytesIO()
#     compressed.write(response.read())
#     compressed.seek(0)
#     decompressed = GzipFile(fileobj=compressed, mode='rb')
#     with open(filename, "wb") as f:
#         f.write(decompressed.read())
#     return filename


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

    legal_atoms = elem.symbols() # element #1-92 (H - U)
    elements, coords = [],[]
    rescount = 0
    with open(pdb_file) as f:
        for line in f:
            if (line[:6] == 'ENDMDL') and not multimodel:
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_label = line[76:78].lstrip().upper()
                (occ, tag) = (float(line[56:60]), line[16])
                use_atom = (occ > 0.5) | ((occ == 0.5) & (tag.upper() == "A"))
                if use_atom and (atom_label in legal_atoms):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    elements.append(elem.number(atom_label))
                    coords.append([x, y, z])
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

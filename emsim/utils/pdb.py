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
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .. import elem
from .. import atom_list as al


# Most amino acids are composed only by C, H, O, N.
# Exceptions: CYS, MET have S. SEC has Se
STANDART_AMINO_ACIDS22 = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CSC', 'LYS', 'MET', 'PHE', 'PRO', 'PYL', 'GLN',
    'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'SEC', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AMIBIGUOUS_AMINO_ACIDS = ['GLX', 'ASX', 'UNK']  # GLX = GLU or GLN, ASX = ASN or ASP, UNK = UNKNOWN
STANDART_NUCLEOTIDES11 = ['A', 'C', 'G', 'I', 'U', 'DA', 'DC', 'DG', 'DI', 'DT', 'DU', 'N']  # N = UNKNOWN

# Te standard residues are recored as ATOM record in PDB or PDBx/mmCIF
STANDART_RESIDUES = STANDART_AMINO_ACIDS22 + AMIBIGUOUS_AMINO_ACIDS + STANDART_NUCLEOTIDES11


def fetch_pdb_file(pdb_code: str, pdir: Union[Path, str] = Path('.'),
                   file_format: str = 'mmCif',
                   overwrite: bool = False) -> str:
    """

    Parameters
    ----------
    pdb_code: str
        the PDB code to download
    pdir: the location to store the dowloaded file
    file_format: str
        {'pdb' | 'mmCif'}
    overwrite: bool
        specifies whether to overwrite existing files

    Returns
    -------
    filename: str

    """
    pdbl = PDBList(server="ftp://ftp.wwpdb.org")
    return pdbl.retrieve_pdb_file(pdb_code=pdb_code, pdir=pdir, file_format=file_format, overwrite=overwrite)


def build_biological_unit(pdb_file: Union[str, Path]):
    atml = read_atoms(pdb_file)
    op = read_symmetries(pdb_file)
    atml = apply_symmetry(atml, op)
    return atml


def fetch_all_pdb_file(pdir):
    pass


def read_atoms(pdb_file: Union[str, Path], identifier: Optional[str] = None) -> al.AtomList:
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
    _, code, parser = check_file_format(pdb_file, make_parser=True)

    if identifier is None:
        identifier = code

    structure = parser.get_structure(identifier, pdb_file)

    elems = []
    coords = []
    # for model in structure:
    #     for chain in model:
    #         for residule in chain:
    #             for a in residule:
    #                 z = elem.number(a.element)
    #                 if z is None:
    #                     continue
    #                 elems.append(elem.number(a.element))
    #                 coords.append(a.get_coord())

    for a in structure.get_atoms():
        z = elem.number(a.element)
        if z is None:
            continue
        elems.append(elem.number(a.element))
        coords.append(a.get_coord())

    return al.AtomList(elements=np.array(elems, dtype=np.int), coordinates=np.array(coords, dtype=np.float))


def check_file_format(pdb_file: Union[str, Path], make_parser: bool = False):
    pdb_file = Path(pdb_file)
    file_format = ''
    code = pdb_file.stem
    # TODO: reliable check needs to peek into the file
    if pdb_file.stem.startswith('pdb') and pdb_file.suffix == '.ent':
        code = code[3:]
        file_format = 'pdb'
    elif pdb_file.suffix == '.cif':
        file_format = 'mmcif'

    if make_parser:
        if file_format == 'pdb':
            parser = PDBParser(PERMISSIVE=True, QUIET=True)
        elif file_format == 'mmcif':
            parser = MMCIFParser(QUIET=True)
        else:
            raise ValueError(f'parser does not support the file format: {str(pdb_file)}')
        return file_format, code, parser
    else:
        return file_format, code


def read_symmetries(pdb_file: Union[str, Path]):
    file_format, code = check_file_format(pdb_file)

    operations = None
    if file_format == 'pdb':
        operations = read_pdb_remark_350(pdb_file)
    elif file_format == 'mmcif':
        operations = read_mmcif_struct_oper_list(pdb_file)
    return operations


def read_mmcif_struct_oper_list(pdb_file: Union[str, Path]) -> np.ndarray:
    """
    reads symmetry operation from a mmCif file from the entry _struct_oper_list.

    The _pdbx_struct_oper_list may contain:
      a 'P' operation to transform the deposited coordinates to a standard point frame
      a 'X0' operation to move the deposited coordinates into the crystal frame
    We are not interested in these two operations here. So we filter out them.
    Here we only find out the operations identified by id enumerated as 1, 2, 3 ...

    Reference: https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies

    Parameters
    ----------
    pdb_file: Union[str, Path]

    Returns
    -------
    3D array in shape (n_symmetries, 3, 4)
        symmetry operators stored in 3 by 4 matrices. The first 3 columns are rotation, the last is translation.
        It at least contains the identity.
    """

    mmcif_dict = MMCIF2Dict(pdb_file)
    op_ids = mmcif_dict["_pdbx_struct_oper_list.id"]
    index_needed = [i for i, d in enumerate(op_ids) if d.strip().isdigit()]

    matrices = []
    for i in range(1, 4):
        rows = []
        b = [float(x) for i, x in enumerate(mmcif_dict[f"_pdbx_struct_oper_list.vector[{i:d}]"]) if i in index_needed]
        for j in range(1, 4):
            rows.append([float(x) for i, x in enumerate(mmcif_dict[f"_pdbx_struct_oper_list.matrix[{i:d}][{j:d}]"]) if i in index_needed])
        rows.append(b)
        matrices.append(rows)
    operations = np.array(matrices)  # (3, 4, n)
    operations = np.transpose(operations, (2, 0, 1))
    return operations


def read_pdb_remark_350(pdb_file: Union[str, Path]) -> np.ndarray:
    """
    gets symmetry operators from a pdb file REMARK 350 section.
    This section contains all operations needed to generate the biomolecule.
    See https://www.wwpdb.org/documentation/file-format-content/format23/remarks2.html for details.

    Parameters
    ----------
    pdb_file: Union[str, Path]

    Returns
    -------
    3D array in shape (n_symmetries, 3, 4)
        symmetry operators stored in 3 by 4 matrices. The first 3 columns are rotation, the last is translation.
        It at least contains the identity.

    Notes
    -----
        The first symmetry operation is identity in normal pdb files, followed by non-trivial symmetries.
    """
    symm_list = []
    with open(pdb_file) as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('REMARK 350') and line[13:18] == "BIOMT":
                symm_list.append([float(x) for x in [line[24:33], line[34:43], line[44:53], line[58:68]]])

    if len(symm_list) == 0:
        # at least add the identity
        symms = np.hstack((np.identity(3), [[0], [0], [0]]))[np.newaxis, :]
    else:
        symms = np.array(symm_list).reshape((-1, 3, 4))
    return symms


def apply_symmetry(atom_list: al.AtomList, operations: np.ndarray):
    """
    applies symmetry operators on atoms to generate whole structure of pdb molecules.
    Here the symmetries must be in shape (n_opers, 3, 4), meaning that the rotation matrices and
    translation vectors are put together.

    Parameters
    ---------
    atom_list: AtomList
        specifies a list of atoms by their Z and the corresponding xyz coordinates
    operations: numpy array
        symmetry operators (integrated rotation and translation) in shape (n_opers, 3, 4)

    Returns
    -------
    A new AtomList object.
        The `elements` array remains the same as the input one, i.e. in shape (n_elems,),
        while the `coordinates`' shape becomes (n_opers, n_elems, 3).
        The correspondence between `elements[i]` and coordinates[:, i, 3] is iterpreted as broadcasting.
    """

    n_opers = operations.shape[0]
    n_atoms = len(atom_list.elements)

    # append the coordinates with ones, so as to perform rotation and translation simultaneously.
    tmp_atom_xyz = np.append(atom_list.coordinates, np.ones((n_atoms, 1)), axis=1)  # (n_atoms, 4)
    # (n_opers, 1, 3, 4) @ (1, n_atoms, 4, 1) -> (n_opers, n_atoms, 3, 1)
    all_atom_xyz = np.matmul(operations[:, None, :, :], tmp_atom_xyz[None, :, :, None])  # (n_opers, n_atoms, 3, 1)
    all_atom_xyz = all_atom_xyz.squeeze().reshape(-1, 3)  # (n_opers * n_atoms, 3)

    all_elems = np.concatenate([atom_list.elements] * n_opers)  # (o_pers * n_atoms, )
    return al.AtomList(elements=all_elems, coordinates=all_atom_xyz)

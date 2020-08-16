import os
from pathlib import Path
import configparser

from .define import default_config_file


def get_global_data_dir_from_config(config_file=None):
    if config_file is None:
        config_file = default_config_file

    config = configparser.ConfigParser()
    config.read(config_file)
    data_dir = Path(config['DEFAULT']['global_data_dir']).resolve()
    if not data_dir.is_dir():
        os.makedirs(data_dir)
        print(f"emsim data dir is created @ {str(pdb_data_dir)}")

    return data_dir


def get_pdb_data_dir_from_config(config_file=None):
    if config_file is None:
        config_file = default_config_file

    config = configparser.ConfigParser()
    config.read(config_file)
    pdb_data_dir = Path(config['DEFAULT']['pdb_data_dir']).resolve()
    if not pdb_data_dir.is_dir():
        os.makedirs(pdb_data_dir)
        print(f"pdb-style data dir is created @ {str(pdb_data_dir)}")

    return pdb_data_dir



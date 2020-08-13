import configparser
from pathlib import Path

from .define import default_config_file


def generate_config_file(config_file_path=None):
    default_data_dir = str(Path.home() / 'emsimData')
    default_pdb_data_dir = str(Path.home() / 'emsimData/pdb')
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'global_data_dir': default_data_dir,
        'pdb_data_dir': default_pdb_data_dir
    }

    if config_file_path is None:
        config_file_path = Path(default_config_file).parent
    else:
        config_file_path = Path(config_file_path)

    with open(config_file_path / "emsimConfig.ini", 'w') as f:
        config.write(f)

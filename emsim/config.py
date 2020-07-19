import configparser
from pathlib import Path

config = configparser.ConfigParser()
config['DEFAULT'] = {
    'data_dir': Path.home() / 'emsimData'
}


with open(Path.home() / 'emsimConfig.ini', 'w') as f:
    config.write(f)
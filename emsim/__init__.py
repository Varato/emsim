from . import config
from . import utils
from . import elem
from . import atoms
from . import em
from . import pipe
from . import simulator
from . import io
from . import pot
from . import wave

import logging as _logging

global_logger = _logging.getLogger(__name__)

ch = _logging.StreamHandler()
ch.setLevel(_logging.INFO)
ch.setFormatter(_logging.Formatter('%(asctime)s::%(levelname)s::%(message)s'))

global_logger.addHandler(ch)
global_logger.setLevel(_logging.INFO)

io.config_file.generate_config_file()

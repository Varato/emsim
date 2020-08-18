from . import config
from . import utils
from . import elem
from . import atoms
from . import em
from . import pipe
from . import simulator
from . import io

import logging as _logging

_logging.basicConfig(level=_logging.INFO)

io.config_file.generate_config_file()

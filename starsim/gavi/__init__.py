import logging
import sys

import pathlib
rootdir = pathlib.Path(__file__).parent.parent.parent
datadir = rootdir / "data"

logger = logging.getLogger("gavi-outbreaks")
debug_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
info_handler = logging.StreamHandler(sys.stdout)  # info_handler will handle all messages below WARNING sending them to STDOUT
warning_handler = logging.StreamHandler(sys.stderr)  # warning_handler will send all messages at or above WARNING to STDERR

debug_handler.setLevel(0)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
info_handler.setLevel(logging.INFO)  # Handle all lower levels - the output should be filtered further by setting the logger level, not the handler level
warning_handler.setLevel(logging.WARNING)

debug_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.INFO})())  # Display anything INFO or higher
info_handler.addFilter(type("ThresholdFilter", (object,), {"filter": lambda x, logRecord: logRecord.levelno < logging.WARNING})())  # Don't display WARNING or higher

debug_formatter = logging.Formatter("%(levelname)s {%(filename)s:%(lineno)d} - %(message)s")
debug_handler.setFormatter(debug_formatter)

logger.addHandler(debug_handler)
logger.addHandler(info_handler)
logger.addHandler(warning_handler)
logger.setLevel("INFO")  # Set the overall log level

# from .analyze_clusters import *
from gavi.interventions import *
from gavi.utils import *
from gavi.ebola import *
from gavi.ebola_main import *
from gavi.multisim import *
from gavi.networks import *

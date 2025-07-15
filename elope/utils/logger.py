
import logging 
from colorlog import ColoredFormatter

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    "%(log_color)s[%(levelname)s]\033[0m %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
))

LOGGER = logging.getLogger("colored_logger")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(handler)
LOGGER.propagate = False  # Prevent double logging

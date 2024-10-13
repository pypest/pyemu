"""Utilities to help with logging."""

import logging
import sys
from typing import Optional

FILE_HANDLER = logging.FileHandler("pyemu.log", delay=True)
STREAM_HANDLER = logging.StreamHandler(sys.stdout)

LOGGER: Optional[logging.Logger] = None


def set_logger(logger: logging.Logger) -> None:
    """Set the global logger to be used by pyemu.

    Args:
        logger (logging.Logger): the logger to be used.
    """
    global LOGGER
    LOGGER = logger


def get_logger(
    name: Optional[str] = "pyemu",
    verbose: bool = False,
    logfile: bool = False,
) -> logging.Logger:
    """Get a logger instance.

    Used to either get
    Args:
        name (`str`): name of the logger (default: "pyemu")
        logger (`bool` or `logging.Logger`): either a boolean indicating to write to
            "pyemu.log" or a logger to return as is.

    Returns:
        logging.Logger object
    """
    if LOGGER is not None:
        return LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    _toggle_handler(logfile, logger, FILE_HANDLER)
    _toggle_handler(verbose, logger, STREAM_HANDLER)
    return logger


def _toggle_handler(switch: bool, logger: logging.Logger, handler: logging.Handler):
    if switch and handler not in logger.handlers:
        logger.addHandler(handler)
    if not switch and handler in logger.handlers:
        logger.removeHandler(handler)

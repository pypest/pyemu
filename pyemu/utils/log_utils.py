"""Utilities to help with logging."""

import logging
import sys
from typing import Optional, Union

FILE_LOGGER = logging.FileHandler("pyemu.log", delay=True)
STREAM_LOGGER = logging.StreamHandler(sys.stdout)


def get_logger(
    name: Optional[str] = "pyemu",
    verbose: bool = False,
    logger: Union[bool, logging.Logger] = True,
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
    if isinstance(logger, bool):
        create_file = logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if create_file is True and FILE_LOGGER not in logger.handlers:
            logger.addHandler(FILE_LOGGER)
    if verbose and STREAM_LOGGER not in logger.handlers:
        logger.addHandler(STREAM_LOGGER)
    if not verbose and STREAM_LOGGER in logger.handlers:
        logger.removeHandler(STREAM_LOGGER)
    return logger

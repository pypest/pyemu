"""Utilities to help with logging."""

import logging
import sys
from typing import Any, Optional, Union

FILE_HANDLER = logging.FileHandler("pyemu.log", delay=True)
STREAM_HANDLER = logging.StreamHandler(sys.stdout)

LOGGER: Optional[logging.Logger] = None


def set_logger(
    logger: Optional[logging.Logger] = None,
    /,
    **basic_config: Any,
) -> None:
    """Set the global logger to be used by pyemu.

    Args:
        logger (logging.Logger): the logger to be used.
        **basic_config (Any): keyword arguments to `logging.basicConfig`
    """
    global LOGGER
    if logger is not None:
        if basic_config:
            raise ValueError(
                "If a logger is passed no extra keyword arguments should be passed as well."
            )
    else:
        if not basic_config:
            raise ValueError(
                "If no logger is passed then keyword arguments for logging.basicConfig should be passed."
            )
        logging.basicConfig(**basic_config)
        logger = logging.getLogger("pyemu")
    LOGGER = logger


def get_logger(
    name: Optional[str] = "pyemu",
    verbose: bool = False,
    logfile: Union[bool , str] = False,
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

    for handler in logger.handlers:
        logger.removeHandler(handler)
    if logfile is True:
        logger.addHandler(FILE_HANDLER)
    elif isinstance(logfile, str):
        logger.addHandler(logging.FileHandler(logfile))

    if verbose:
        logger.addHandler(STREAM_HANDLER)

    return logger

import logging

"""
Utilities for more useful logging output.
"""

import logging
import traceback
from typing import Union

import numpy as np

FORMAT_PREFIX = (
    "%(levelname).1s%(asctime)s.%(msecs)d %(process)d %(filename)s:%(lineno)d]"
)
FORMAT = f"{FORMAT_PREFIX} %(message)s"
DATEFMT = "%m%d %H:%M:%S"


# Helper for things that format as brackets around comma-separated items
def _seq_repr(parts, prefix, suffix, indent):
    if len(parts) <= 1 or sum(len(p) for p in parts) < 80:
        return prefix + ", ".join(parts) + suffix
    return (
        f"{prefix}\n{indent}  " + f",\n{indent}  ".join(parts) + f"\n{indent}{suffix}"
    )


class LocalsFormatter(logging.Formatter):
    """
    logging.Formatter which shows local variables in stack dumps.
    """

    def formatException(self, exc_info) -> str:
        tb = traceback.TracebackException(*exc_info, capture_locals=True)
        # try not to be too verbose
        for frame in tb.stack:
            if len(repr(frame.locals)) > 500:
                frame.locals = None
        return "".join(tb.format())


def config_logging(
    *,
    fmt: str = FORMAT,
    level: Union[int, str] = logging.INFO,
    datefmt: str = DATEFMT,
    style: str = "%",
    stream=None,
) -> None:
    """
    Configure logging.

    Same as `logging.basicConfig(fmt, level, datefmt, style, force=True)`,
    except it uses :class:`LocalsFormatter` to show local variables in
    exception stack traces.

    stream: If specified, the root logger will use it for logging output; otherwise,
        sys.stderr will be used.
    """
    root = logging.getLogger()

    # always do 'force'
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    h = logging.StreamHandler(stream)
    h.setFormatter(LocalsFormatter(fmt, datefmt, style))
    root.addHandler(h)
    root.setLevel(level)


DEBUG: int = logging.DEBUG
INFO: int = logging.INFO
WARNING: int = logging.WARNING
ERROR: int = logging.ERROR

WHITE = "\x1b[37;20m"
WHITE_BOLD = "\x1b[37;1m"
BLUE = "\x1b[34;20m"
BLUE_BOLD = "\x1b[34;1m"
RESET = "\x1b[0m"

Logger = logging.Logger


def get_logger(level: int = logging.INFO) -> Logger:
    """Provides a logger with the specified logging level.

    Returns:
        A logger.
    """

    config_logging(level=level)
    return logging.getLogger(__name__)


def get_separator(length: int = 80) -> str:
    """Return a text separator to be used in logs.

    Args:
        length: Length of the separator (in the number of characters).
    """

    return length * "-"


def log_heading(logger: Logger, msg: str, style: str = WHITE) -> None:
    """Logs a visually distinct heading.

    Args:
        logger: A logger.
        heading: The heading to print.
    """

    separator = get_separator()
    logger.info(style + separator + RESET)
    logger.info(style + msg + RESET)
    logger.info(style + separator + RESET)

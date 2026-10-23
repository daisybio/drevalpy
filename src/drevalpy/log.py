"""Centralized logging for drevalpy using Rich.

Importing this module only attaches a :class:`logging.NullHandler` to the ``drevalpy`` logger, as is
conventional for libraries. Applications call :func:`setup_logging` to get Rich output.
"""

import logging

from rich.logging import RichHandler

_FORMAT = "%(message)s"
_LOG_LEVEL = logging.INFO

logging.getLogger("drevalpy").addHandler(logging.NullHandler())


def setup_logging(level: int = _LOG_LEVEL) -> None:
    """Configure the drevalpy logger with Rich output.

    Call this once at application startup. Subsequent calls update the level and do not add a
    second Rich handler.

    :param level: Logging level (default: INFO).
    """
    logger = logging.getLogger("drevalpy")
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the drevalpy namespace.

    :param name: Module name (typically ``__name__``).
    :returns: Logger instance.
    """
    return logging.getLogger(name)

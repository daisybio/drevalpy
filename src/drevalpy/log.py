"""Centralized logging for drevalpy using Rich."""

import logging

from rich.logging import RichHandler

_FORMAT = "%(message)s"
_LOG_LEVEL = logging.INFO


def setup_logging(level: int = _LOG_LEVEL) -> None:
    """Configure the drevalpy logger with Rich output.

    Call this once at application startup. Subsequent calls update the level.

    :param level: Logging level (default: INFO).
    """
    logger = logging.getLogger("drevalpy")
    if not logger.handlers:
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


setup_logging()

"""Pytest configuration for the CLI tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_drevalpy_logger() -> Iterator[None]:
    """Undo the Rich handler and log level the root callback sets up through ``setup_logging``."""
    logger = logging.getLogger("drevalpy")
    handlers = list(logger.handlers)
    level = logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)

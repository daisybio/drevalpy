"""Pytest configuration for the CLI tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest


def _restoring_drevalpy_logger() -> Iterator[None]:
    logger = logging.getLogger("drevalpy")
    handlers = list(logger.handlers)
    level = logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture(autouse=True)
def _restore_drevalpy_logger() -> Iterator[None]:
    """Undo the Rich handler and log level the root callback sets up through ``setup_logging``."""
    yield from _restoring_drevalpy_logger()


@pytest.fixture(autouse=True, scope="module")
def _restore_drevalpy_logger_after_module() -> Iterator[None]:
    """Do the same for module- and class-scoped fixtures, which run before the per-test snapshot."""
    yield from _restoring_drevalpy_logger()


@pytest.fixture(scope="session")
def dataset_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Write the small synthetic dataset to an ``.h5mu`` file once and return its path."""
    from drevalpy.data import synthetic

    path = tmp_path_factory.mktemp("cli_dataset") / "toy.h5mu"
    synthetic().save(path)
    return str(path)

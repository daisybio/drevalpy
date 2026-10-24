"""Tests for the centralized Rich-backed logging setup.

Pytest and other tests touch handlers too, so every test restores the ``drevalpy`` logger and
asserts on that logger only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from drevalpy.log import get_logger, setup_logging
from tests._trusted_subprocess import run_trusted_python


@pytest.fixture
def drevalpy_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("drevalpy")
    handlers = list(logger.handlers)
    level = logger.level
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_get_logger_matches_the_standard_library_and_propagates(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("drevalpy.models.foo")

    assert logger is logging.getLogger("drevalpy.models.foo")
    with caplog.at_level(logging.INFO, logger="drevalpy"):
        logger.info("hello")
    assert "hello" in caplog.text


def test_import_attaches_no_rich_handler() -> None:
    script = (
        "import logging, drevalpy\n"
        "handlers = logging.getLogger('drevalpy').handlers\n"
        "assert not any(type(h).__name__ == 'RichHandler' for h in handlers), handlers\n"
        "assert any(isinstance(h, logging.NullHandler) for h in handlers), handlers\n"
    )

    result = run_trusted_python(script)

    assert result.returncode == 0, result.stderr


class TestSetupLogging:
    def test_attaches_a_configured_rich_handler_once(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.handlers[:] = []
        drevalpy_logger.setLevel(logging.CRITICAL)

        setup_logging()
        existing = list(drevalpy_logger.handlers)
        setup_logging(logging.WARNING)

        handler = existing[0]
        assert [type(h) for h in existing] == [RichHandler]
        assert drevalpy_logger.handlers == existing
        assert handler.markup is True
        assert handler.rich_tracebacks is True
        record = logging.LogRecord("drevalpy.x", logging.INFO, "f.py", 1, "plain text", None, None)
        assert handler.formatter is not None
        assert handler.formatter.format(record) == "plain text"
        assert drevalpy_logger.level == logging.WARNING

    def test_defaults_to_info(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.setLevel(logging.CRITICAL)

        setup_logging()

        assert drevalpy_logger.level == logging.INFO

    def test_keeps_the_null_handler_next_to_the_rich_handler(self, drevalpy_logger: logging.Logger) -> None:
        sentinel = logging.NullHandler()
        drevalpy_logger.handlers[:] = [sentinel]

        setup_logging(logging.ERROR)

        assert [type(h) for h in drevalpy_logger.handlers] == [logging.NullHandler, RichHandler]
        assert drevalpy_logger.handlers[0] is sentinel
        assert drevalpy_logger.level == logging.ERROR

    def test_emitted_records_reach_handlers_at_the_configured_level(
        self, drevalpy_logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        setup_logging(logging.WARNING)

        with caplog.at_level(logging.DEBUG, logger="drevalpy"):
            get_logger("drevalpy.models.foo").warning("warned")

        assert [r.message for r in caplog.records] == ["warned"]

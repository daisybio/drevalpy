"""Tests for the centralized Rich-backed logging setup.

``drevalpy.log`` calls :func:`~drevalpy.log.setup_logging` at import time, so the
``drevalpy`` logger is already configured before any test runs, and pytest adds
handlers of its own to the *root* logger. Every test here therefore restores the
``drevalpy`` logger it touches and asserts on that logger only, never on global
handler counts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from drevalpy.log import get_logger, setup_logging


@pytest.fixture
def drevalpy_logger() -> Iterator[logging.Logger]:
    """Yield the package logger, restoring its handlers and level afterwards."""
    logger = logging.getLogger("drevalpy")
    handlers = list(logger.handlers)
    level = logger.level
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)


class TestGetLogger:
    def test_returns_a_logger_with_the_requested_name(self) -> None:
        assert get_logger("drevalpy.models.foo").name == "drevalpy.models.foo"

    def test_is_idempotent_for_one_name(self) -> None:
        assert get_logger("drevalpy.models.foo") is get_logger("drevalpy.models.foo")

    def test_matches_the_standard_library_lookup(self) -> None:
        assert get_logger("drevalpy.models.foo") is logging.getLogger("drevalpy.models.foo")

    def test_child_loggers_propagate_to_the_package_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("drevalpy.models.foo")

        with caplog.at_level(logging.INFO, logger="drevalpy"):
            logger.info("hello")

        assert "hello" in caplog.text


class TestSetupLogging:
    def test_import_time_call_already_attached_a_rich_handler(self) -> None:
        logger = logging.getLogger("drevalpy")

        assert any(isinstance(h, RichHandler) for h in logger.handlers)

    def test_attaches_a_rich_handler_when_none_is_present(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.handlers[:] = []

        setup_logging()

        assert [type(h) for h in drevalpy_logger.handlers] == [RichHandler]

    def test_configured_handler_enables_markup_and_rich_tracebacks(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.handlers[:] = []

        setup_logging()

        handler = drevalpy_logger.handlers[0]
        assert isinstance(handler, RichHandler)
        assert handler.markup is True
        assert handler.rich_tracebacks is True

    def test_configured_handler_formats_the_bare_message(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.handlers[:] = []

        setup_logging()

        formatter = drevalpy_logger.handlers[0].formatter
        assert formatter is not None
        assert (
            formatter.format(logging.LogRecord("drevalpy.x", logging.INFO, "f.py", 1, "plain text", None, None))
            == "plain text"
        )

    def test_defaults_to_info(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.setLevel(logging.CRITICAL)

        setup_logging()

        assert drevalpy_logger.level == logging.INFO

    def test_honours_an_explicit_level(self, drevalpy_logger: logging.Logger) -> None:
        setup_logging(logging.DEBUG)

        assert drevalpy_logger.level == logging.DEBUG

    def test_does_not_add_a_second_handler(self, drevalpy_logger: logging.Logger) -> None:
        drevalpy_logger.handlers[:] = []
        setup_logging()
        existing = list(drevalpy_logger.handlers)

        setup_logging(logging.WARNING)

        assert drevalpy_logger.handlers == existing

    def test_updates_the_level_of_an_already_configured_logger(self, drevalpy_logger: logging.Logger) -> None:
        sentinel = logging.NullHandler()
        drevalpy_logger.handlers[:] = [sentinel]

        setup_logging(logging.ERROR)

        assert drevalpy_logger.handlers == [sentinel]
        assert drevalpy_logger.level == logging.ERROR

    def test_emitted_records_reach_handlers_at_the_configured_level(
        self, drevalpy_logger: logging.Logger, caplog: pytest.LogCaptureFixture
    ) -> None:
        setup_logging(logging.WARNING)

        with caplog.at_level(logging.DEBUG, logger="drevalpy"):
            get_logger("drevalpy.models.foo").warning("warned")

        assert [r.message for r in caplog.records] == ["warned"]

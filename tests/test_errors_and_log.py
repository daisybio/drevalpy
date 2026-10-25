"""Tests for the ``DrEvalError`` hierarchy and the Rich-backed logging setup."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from drevalpy import errors
from drevalpy.errors import DrEvalError, InputNotFoundError, InvalidInputError, UnknownNameError
from drevalpy.log import get_logger, setup_logging
from tests._trusted_subprocess import run_trusted_python


@pytest.mark.parametrize(
    ("error_class", "builtins"),
    [
        (UnknownNameError, (ValueError, KeyError)),
        (InputNotFoundError, (FileNotFoundError,)),
        (InvalidInputError, (ValueError,)),
    ],
)
def test_each_class_keeps_the_builtin_it_replaces(error_class: type[DrEvalError], builtins: tuple[type, ...]) -> None:
    error = error_class("message")

    assert isinstance(error, DrEvalError)
    assert all(isinstance(error, builtin) for builtin in builtins)


@pytest.mark.parametrize("error_class", [UnknownNameError, InputNotFoundError, InvalidInputError])
def test_str_is_the_plain_message(error_class: type[DrEvalError]) -> None:
    assert str(error_class("Unknown model: 'x'")) == "Unknown model: 'x'"


def test_unknown_name_error_is_not_quoted_like_a_key_error() -> None:
    assert str(KeyError("x")) == "'x'"
    assert str(UnknownNameError("x")) == "x"


def test_several_arguments_fall_back_to_the_builtin_text() -> None:
    assert str(InvalidInputError("a", "b")) == str(ValueError("a", "b"))


def test_existing_except_clauses_still_catch_the_new_classes() -> None:
    with pytest.raises(KeyError):
        raise UnknownNameError("x")
    with pytest.raises(FileNotFoundError):
        raise InputNotFoundError("x")
    with pytest.raises(ValueError):
        raise InvalidInputError("x")


def test_public_surface() -> None:
    assert sorted(errors.__all__) == ["DrEvalError", "InputNotFoundError", "InvalidInputError", "UnknownNameError"]


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

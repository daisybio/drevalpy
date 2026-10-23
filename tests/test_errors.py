"""Tests for the DrEvalError hierarchy."""

from __future__ import annotations

import pytest

from drevalpy import errors
from drevalpy.errors import DrEvalError, InputNotFoundError, InvalidInputError, UnknownNameError


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

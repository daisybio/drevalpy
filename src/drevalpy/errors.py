"""Errors raised for mistakes a user can fix, so the CLI can report them without a traceback."""

from __future__ import annotations

__all__ = ["DrEvalError", "InputNotFoundError", "InvalidInputError", "UnknownNameError"]


class DrEvalError(Exception):
    """Base class of every error drevalpy raises for a problem in the user's input or setup."""

    def __str__(self) -> str:
        """Return the plain message.

        :returns: The single argument as text; a ``KeyError`` base would otherwise quote it.
        """
        if len(self.args) == 1:
            return str(self.args[0])
        return super().__str__()


class UnknownNameError(DrEvalError, ValueError, KeyError):
    """A name that is not registered, such as a model, dataset, splitter or featurizer."""


class InputNotFoundError(DrEvalError, FileNotFoundError):
    """A file or directory the user pointed at does not exist."""


class InvalidInputError(DrEvalError, ValueError):
    """An argument or input file exists but its value cannot be used."""

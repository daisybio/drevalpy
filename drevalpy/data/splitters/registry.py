"""Splitter registry: maps mode names to validated splitter functions.

A splitter is any callable with the signature::

    (mudataset: MuDataLike, n_splits: int, validation_ratio: float, random_state: int) -> list[SplitMasks]

Register custom splitters with the ``@splitter_registry.register`` decorator.
Validation runs automatically after each split -- no way to bypass it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Protocol

from drevalpy.data.structures import MuDataLike, SplitMasks

from .validation import Validation, validate_folds


class Splitter(Protocol):
    """Protocol defining the splitter callable signature."""

    def __call__(
        self,
        mudataset: MuDataLike,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[SplitMasks]:
        """Execute the splitter."""
        ...


def _wrap_with_validation(fn: Splitter, validation: Validation) -> Splitter:
    """Wrap a splitter so validation runs automatically after each call."""

    @wraps(fn)
    def wrapper(
        mudataset: MuDataLike,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[SplitMasks]:
        folds = fn(mudataset, n_splits, validation_ratio, random_state)
        validate_folds(folds, validation, mudataset)
        return folds

    return wrapper  # type: ignore[return-value]


class SplitterRegistry:
    """Registry mapping mode names to validated splitter callables."""

    def __init__(self) -> None:
        """Initialize with an empty registry."""
        self._splitters: dict[str, Splitter] = {}
        self._descriptions: dict[str, str] = {}
        self._validations: dict[str, Validation] = {}

    @property
    def modes(self) -> list[str]:
        """Sorted list of registered mode names."""
        return sorted(self._splitters)

    def register(self, mode: str, description: str, validation: Validation) -> Callable[[Splitter], Splitter]:
        """Decorator to register a splitter function under a mode name.

        All three parameters are required. The splitter is automatically wrapped
        so that validation runs after every call.

        :param mode: Mode name (e.g. "LPO", "LCO", or a custom name).
        :param description: Human-readable description of the splitting approach.
        :param validation: Which leakage constraint to enforce ("LCO", "LDO", "LPO", "LTO").
        :returns: Decorator that registers and returns the wrapped function.

        Example::

            @splitter_registry.register("MY_LCO", "Custom LCO with fraction", validation="LCO")
            def my_lco(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
                ...
        """

        def decorator(fn: Splitter) -> Splitter:
            wrapped = _wrap_with_validation(fn, validation)
            self._splitters[mode] = wrapped
            self._descriptions[mode] = description
            self._validations[mode] = validation
            return wrapped

        return decorator

    def get(self, mode: str) -> Splitter:
        """Return the validated splitter for the given mode.

        :param mode: Registered mode name.
        :returns: Splitter callable (with validation baked in).
        :raises ValueError: If mode is not registered.
        """
        splitter = self._splitters.get(mode)
        if splitter is None:
            raise ValueError(f"Unknown split mode {mode!r}. Registered: {self.modes}")
        return splitter

    def resolve(self, splitter: str | Splitter) -> Splitter:
        """Resolve a splitter from a mode string or pass through a callable.

        :param splitter: Either a mode name (str) or a Splitter callable.
        :returns: Splitter callable.
        """
        if isinstance(splitter, str):
            return self.get(splitter)
        return splitter

    def describe(self, mode: str) -> str:
        """Return the description for a registered mode.

        :param mode: Registered mode name.
        :returns: Description string.
        """
        return self._descriptions.get(mode, "")

    def __repr__(self) -> str:
        """Return a Rich table of registered splitters."""
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Registered Splitters")
        table.add_column("Mode")
        table.add_column("Description")
        table.add_column("Validation")

        for mode in self.modes:
            table.add_row(mode, self._descriptions.get(mode, ""), self._validations.get(mode, ""))

        console = Console(width=100, highlight=False)
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()


splitter_registry = SplitterRegistry()

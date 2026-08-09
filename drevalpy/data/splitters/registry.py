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


def _wrap_with_validation(fn: Splitter, mode: str, validation: Validation) -> Splitter:
    """Wrap a splitter so validation runs and default metadata is injected."""

    @wraps(fn)
    def wrapper(
        mudataset: MuDataLike,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[SplitMasks]:
        folds = fn(mudataset, n_splits, validation_ratio, random_state)
        validate_folds(folds, validation, mudataset)
        for i, fold in enumerate(folds):
            fold.metadata.setdefault("mode", mode)
            fold.metadata.setdefault("fold_index", i)
            fold.metadata.setdefault("n_splits", n_splits)
            fold.metadata.setdefault("validation_ratio", validation_ratio)
            fold.metadata.setdefault("random_state", random_state)
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
            def my_lco(mudataset, n_splits=5, validation_ratio=0.1, random_state=42): ...
        """

        def decorator(fn: Splitter) -> Splitter:
            wrapped = _wrap_with_validation(fn, mode, validation)
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

    def to_dataframe(self) -> "pd.DataFrame":
        """Return registry contents as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for mode in self.modes:
            rows.append({
                "Mode": mode,
                "Description": self._descriptions.get(mode, ""),
                "Validation": self._validations.get(mode, ""),
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)

    def _repr_html_(self) -> str:
        """HTML table for Jupyter notebooks."""
        rows = ""
        for mode in self.modes:
            rows += (
                f"<tr><td><b>{mode}</b></td>"
                f"<td>{self._descriptions.get(mode, '')}</td>"
                f"<td>{self._validations.get(mode, '')}</td></tr>\n"
            )

        return (
            "<h4>Registered Splitters</h4>\n"
            '<table border="1" style="border-collapse: collapse; width: 100%;">\n'
            "<thead><tr><th>Mode</th><th>Description</th><th>Validation</th></tr></thead>\n"
            f"<tbody>{rows}</tbody>\n"
            "</table>"
        )


splitter_registry = SplitterRegistry()

"""Splitter registry: maps mode names to splitter functions.

A splitter is any callable with the signature::

    (mudataset: MuDataLike, n_splits: int, validation_ratio: float, random_state: int) -> list[SplitMasks]

Register custom splitters with ``splitter_registry.register("MY_MODE", my_func)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from drevalpy.data.structures import MuDataLike, SplitMasks


class Splitter(Protocol):
    """Protocol defining the splitter callable signature."""

    def __call__(
        self,
        mudataset: MuDataLike,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[SplitMasks]: ...


class SplitterRegistry:
    """Registry mapping mode names to splitter callables."""

    def __init__(self) -> None:
        """Initialize with an empty registry."""
        self._splitters: dict[str, Splitter] = {}
        self._descriptions: dict[str, str] = {}

    @property
    def modes(self) -> list[str]:
        """Sorted list of registered mode names."""
        return sorted(self._splitters)

    def register(self, mode: str, description: str = "") -> Callable[[Splitter], Splitter]:
        """Decorator to register a splitter function under a mode name.

        :param mode: Mode name (e.g. "LPO", "LCO", or a custom name).
        :param description: Human-readable description of the splitting strategy.
        :returns: Decorator that registers and returns the function unchanged.

        Example::

            @splitter_registry.register("MY_MODE", description="My custom split")
            def my_splitter(mudataset, n_splits=5, ...):
                ...
        """

        def decorator(fn: Splitter) -> Splitter:
            self._splitters[mode] = fn
            self._descriptions[mode] = description or fn.__doc__ or ""
            return fn

        return decorator

    def get(self, mode: str) -> Splitter:
        """Return the splitter for the given mode.

        :param mode: Registered mode name.
        :returns: Splitter callable.
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
        """Return a summary of registered splitters."""
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Registered Splitters")
        table.add_column("Mode")
        table.add_column("Description")

        for mode in self.modes:
            table.add_row(mode, self._descriptions.get(mode, ""))

        console = Console(width=100, highlight=False)
        with console.capture() as capture:
            console.print(table)
        return capture.get().rstrip()


splitter_registry = SplitterRegistry()

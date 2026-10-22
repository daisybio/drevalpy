"""Splitter registry: maps mode names to validated splitter functions.

A splitter is any callable with the signature::

    (mudataset: MuDataLike, n_splits: int, validation_ratio: float, random_state: int) -> list[SplitMasks]

Register custom splitters with the ``@splitter_registry.register`` decorator.
Validation runs automatically after each split -- no way to bypass it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol

from drevalpy.types import MuDataLike, SplitMasks

from ._validation import Validation, canonical_validation, validate_folds

if TYPE_CHECKING:
    import pandas as pd


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
        self._aliases: dict[str, str] = {}

    @property
    def modes(self) -> list[str]:
        """Sorted list of registered mode names."""
        return sorted(self._splitters)

    def register(
        self,
        mode: str,
        description: str,
        validation: Validation,
        *,
        override: bool = False,
        aliases: tuple[str, ...] = (),
    ) -> Callable[[Splitter], Splitter]:
        """Decorator to register a splitter function under a mode name.

        The first three parameters are required. The splitter is automatically
        wrapped so that validation runs after every call.

        Registering a mode name that is already taken raises, matching the
        visualization registry: a silent overwrite means one package quietly
        changes another package's split semantics. Pass ``override=True`` when
        replacing a mode is the intent.

        :param mode: Mode name (e.g. "LPO", "LSO", or a custom name).
        :param description: Human-readable description of the splitting approach.
        :param validation: Which leakage constraint to enforce ("LSO", "LDO", "LPO", "LTO").
        :param override: Replace an already-registered mode instead of raising.
        :param aliases: Alternative names that resolve to *mode*. Results, metadata and
            file names always use *mode*, never the alias the caller typed.
        :returns: Decorator that registers and returns the wrapped function.
        :raises ValueError: If *mode* is already registered and *override* is false, or
            if an alias collides with another mode or alias.

        Example::

            @splitter_registry.register("MY_LSO", "Custom LSO with fraction", validation="LSO")
            def my_lso(mudataset, n_splits=5, validation_ratio=0.1, random_state=42): ...
        """

        def decorator(fn: Splitter) -> Splitter:
            if mode in self._splitters and not override:
                msg = f"Splitter mode {mode!r} already registered"
                raise ValueError(msg)
            self._check_aliases(mode, aliases)
            wrapped = _wrap_with_validation(fn, mode, validation)
            for alias in self.aliases_of(mode):
                del self._aliases[alias]
            self._splitters[mode] = wrapped
            self._descriptions[mode] = description
            self._validations[mode] = canonical_validation(validation)
            self._aliases.update(dict.fromkeys(aliases, mode))
            return wrapped

        return decorator

    def _check_aliases(self, mode: str, aliases: tuple[str, ...]) -> None:
        """Reject aliases that shadow a mode or belong to another mode."""
        if mode in self._aliases:
            raise ValueError(f"Splitter mode {mode!r} conflicts with an alias of {self._aliases[mode]!r}")
        for alias in aliases:
            if alias == mode or alias in self._splitters:
                raise ValueError(f"Splitter alias {alias!r} conflicts with the registered mode {alias!r}")
            owner = self._aliases.get(alias)
            if owner is not None and owner != mode:
                raise ValueError(f"Splitter alias {alias!r} already registered for mode {owner!r}")

    def canonical(self, mode: str) -> str:
        """Return the canonical mode name for *mode*, resolving an alias.

        Names that are not aliases are returned unchanged, so unknown modes still
        fail later with the usual "Unknown split mode" error.

        :param mode: Mode name or alias.
        :returns: Canonical mode name.
        """
        return self._aliases.get(mode, mode)

    def aliases_of(self, mode: str) -> list[str]:
        """Return the sorted aliases registered for the canonical *mode*."""
        return sorted(alias for alias, target in self._aliases.items() if target == mode)

    def get(self, mode: str) -> Splitter:
        """Return the validated splitter for the given mode or alias.

        :param mode: Registered mode name or alias.
        :returns: Splitter callable (with validation baked in).
        :raises ValueError: If mode is not registered.
        """
        splitter = self._splitters.get(self.canonical(mode))
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
        return self._descriptions.get(self.canonical(mode), "")

    def get_metadata(self, mode: str) -> dict[str, Any]:
        """Return the metadata record for the splitter registered under *mode*.

        Mirrors ``get_metadata`` on the component registries so every registry can
        be introspected the same way.

        :param mode: Registered mode name.
        :returns: Metadata dict for catalog listings.
        :raises ValueError: If *mode* is not registered.
        """
        self.get(mode)
        mode = self.canonical(mode)
        return {
            "registry": "splitters",
            "name": mode,
            "description": self._descriptions.get(mode, ""),
            "validation": self._validations.get(mode, ""),
            "aliases": self.aliases_of(mode),
        }

    def list_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for every registered mode.

        :returns: List of metadata dicts, ordered by mode name.
        """
        return [self.get_metadata(mode) for mode in self.modes]

    def to_dataframe(self) -> pd.DataFrame:
        """Return registry contents as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for mode in self.modes:
            rows.append(
                {
                    "Mode": mode,
                    "Description": self._descriptions.get(mode, ""),
                    "Validation": self._validations.get(mode, ""),
                    "Aliases": ", ".join(self.aliases_of(mode)),
                }
            )
        return pd.DataFrame(rows)

    def retain_only(self, modes: frozenset[str]) -> None:
        """Remove all entries not in the given set (for rollback support).

        :param modes: Set of mode names to keep.
        """
        for mode in list(self._splitters):
            if mode not in modes:
                del self._splitters[mode]
                self._descriptions.pop(mode, None)
                self._validations.pop(mode, None)
        self._aliases = {alias: mode for alias, mode in self._aliases.items() if mode in self._splitters}

    def __repr__(self) -> str:
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)


splitter_registry = SplitterRegistry()

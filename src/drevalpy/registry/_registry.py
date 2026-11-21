"""Generic thread-safe name-to-object registry shared by every component kind.

One :class:`Registry` instance holds one kind of entry - predictor classes,
featurizer classes, splitters, visualizations or zoo presets - under unique names,
in registration order, each with a read-only metadata mapping. A duplicate name
raises unless the caller asks to replace it, every lookup of an unknown name raises
:class:`~drevalpy.errors.UnknownNameError` naming the close matches and every registered
name, and :meth:`Registry.snapshot` /
:meth:`Registry.restore` let tests undo whatever they registered.

The registry neither inspects nor stamps the objects it stores: checks that depend
on what an entry is (abstract methods, a hyperparameter space, the sides a
featurizer supports) belong to the decorator that wraps :meth:`Registry.register`
for that kind. :class:`TableReprMixin` gives every registry, the dataset registry
included, the same console and notebook table.
"""

from __future__ import annotations

import difflib
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from drevalpy.errors import UnknownNameError

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T")

_EMPTY: Mapping[str, Any] = MappingProxyType({})


class TableReprMixin(ABC):
    """Notebook and console representations built from a registry's ``to_dataframe``."""

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Return the registry contents as a pandas DataFrame.

        :returns: One row per registered entry.
        """

    def __repr__(self) -> str:
        """Return a tabular string representation."""
        return self.to_dataframe().to_string(index=False)

    def _repr_html_(self) -> str:
        """Return an HTML table for Jupyter notebooks."""
        return self.to_dataframe().to_html(index=False)


@dataclass(frozen=True)
class _Entry(Generic[T]):
    obj: T
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Snapshot(Generic[T]):
    """The entries of one registry at one moment, for :meth:`Registry.restore`."""

    kind: str
    entries: tuple[tuple[str, T, Mapping[str, Any]], ...]


def _frozen_metadata(kind: str, name: str, metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return *metadata* as a read-only copy after checking its keys.

    :param kind: Registry kind, named in the error message.
    :param name: Entry name, named in the error message.
    :param metadata: Metadata given at registration, or ``None``.
    :returns: Read-only mapping, empty when *metadata* is ``None``.
    :raises TypeError: If a metadata key is not a string.
    """
    if not metadata:
        return _EMPTY
    bad = [key for key in metadata if not isinstance(key, str)]
    if bad:
        msg = f"{kind} {name!r} metadata keys must be strings, got {bad!r}"
        raise TypeError(msg)
    return MappingProxyType(dict(metadata))


def _suggestion(name: str, names: Collection[str]) -> str:
    """Return a "Did you mean" sentence naming the registered names closest to *name*.

    Case is ignored, so ``scaledgeneexpression`` finds ``scaledGeneExpression``.

    :param name: The unknown name.
    :param names: The registered names.
    :returns: ``"Did you mean 'a' or 'b'? "``, or ``""`` when no name is close.
    """
    by_lower = {candidate.lower(): candidate for candidate in names}
    matches = difflib.get_close_matches(name.lower(), by_lower, n=3, cutoff=0.6)
    if not matches:
        return ""
    quoted = [repr(by_lower[match]) for match in matches]
    options = quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"
    return f"Did you mean {options}? "


def _cell(value: Any) -> Any:
    """Render one metadata value as a table cell.

    :param value: Metadata value.
    :returns: Sets joined in sorted order, ``""`` for ``None``, anything else unchanged.
    """
    if value is None:
        return ""
    if isinstance(value, (set, frozenset)):
        return ", ".join(sorted(str(item) for item in value))
    return value


class Registry(TableReprMixin, Generic[T]):
    """Thread-safe store of named entries of one kind, in registration order."""

    def __init__(self, kind: str) -> None:
        """Create an empty registry.

        :param kind: What the entries are, such as ``"predictor"``; used in error messages.
        """
        self.kind = kind
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        obj: T,
        *,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> T:
        """Store *obj* under *name*.

        :param name: Unique, non-empty entry name.
        :param obj: Entry to store.
        :param metadata: Read-only facts about the entry, such as ``description`` and ``tags``.
        :param replace: Overwrite an entry already registered under *name* instead of raising.
        :returns: *obj*, so the call can end a decorator.
        :raises ValueError: If *name* is empty, or already registered and *replace* is false.
        """
        if not isinstance(name, str) or not name.strip():
            msg = f"{self.kind} name must be a non-empty string, got {name!r}"
            raise ValueError(msg)
        entry = _Entry(obj, _frozen_metadata(self.kind, name, metadata))
        with self._lock:
            if name in self._entries and not replace:
                msg = f"{self.kind} {name!r} already registered"
                raise ValueError(msg)
            self._entries[name] = entry
        return obj

    def decorator(self, name: str, **metadata: Any) -> Callable[[T], T]:
        r"""Return a decorator that registers its target under *name*.

        :param name: Unique entry name.
        :param \**metadata: Metadata stored with the entry.
        :returns: Decorator that registers and returns the decorated object.
        """

        def decorate(obj: T) -> T:
            return self.register(name, obj, metadata=metadata)

        return decorate

    def get(self, name: str) -> T:
        """Return the entry registered under *name*.

        An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

        :param name: Entry name.
        :returns: The registered object.
        """
        return self._entry(name).obj

    def _entry(self, name: str) -> _Entry[T]:
        """Return the stored entry for *name*.

        :param name: Entry name.
        :returns: The object and its metadata.
        :raises UnknownNameError: If *name* is not registered.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                available = ", ".join(self._entries) or "<none>"
                msg = f"Unknown {self.kind} {name!r}. {_suggestion(name, self._entries)}Available: {available}"
                raise UnknownNameError(msg)
            return entry

    def names(self) -> tuple[str, ...]:
        """Return the registered names.

        :returns: Names in registration order.
        """
        with self._lock:
            return tuple(self._entries)

    def metadata(self, name: str) -> Mapping[str, Any]:
        """Return the metadata registered with *name*.

        An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

        :param name: Entry name.
        :returns: Read-only metadata mapping, empty when none was given.
        """
        return self._entry(name).metadata

    def items(self) -> tuple[tuple[str, T], ...]:
        """Return every entry with its name.

        :returns: ``(name, obj)`` pairs in registration order.
        """
        with self._lock:
            return tuple((name, entry.obj) for name, entry in self._entries.items())

    def __contains__(self, name: object) -> bool:
        """Return whether *name* is registered.

        :param name: Candidate entry name.
        :returns: ``True`` when an entry is registered under *name*.
        """
        with self._lock:
            return name in self._entries

    def __len__(self) -> int:
        """Return the number of entries.

        :returns: How many names are registered.
        """
        with self._lock:
            return len(self._entries)

    def snapshot(self) -> Snapshot[T]:
        """Capture the current entries.

        :returns: Snapshot that :meth:`restore` puts back.
        """
        with self._lock:
            entries = tuple((name, entry.obj, entry.metadata) for name, entry in self._entries.items())
        return Snapshot(self.kind, entries)

    def restore(self, snapshot: Snapshot[T]) -> None:
        """Replace the entries with those captured by :meth:`snapshot`.

        :param snapshot: Snapshot taken from a registry of the same kind.
        :raises ValueError: If *snapshot* was taken from a registry of another kind.
        """
        if snapshot.kind != self.kind:
            msg = f"cannot restore a {snapshot.kind} snapshot into the {self.kind} registry"
            raise ValueError(msg)
        with self._lock:
            self._entries = {name: _Entry(obj, metadata) for name, obj, metadata in snapshot.entries}

    def to_dataframe(self) -> pd.DataFrame:
        """Return the entries as a table.

        :returns: One row per entry: a ``Name`` column, then one column per metadata key.
        """
        import pandas as pd

        with self._lock:
            rows = [
                {
                    "Name": name,
                    **{key.replace("_", " ").capitalize(): _cell(value) for key, value in entry.metadata.items()},
                }
                for name, entry in self._entries.items()
            ]
        return pd.DataFrame(rows, columns=None if rows else ["Name"])

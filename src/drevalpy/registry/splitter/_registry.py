"""Splitter registry: maps mode names to validated splitter functions.

A splitter is any callable with the signature::

    (dataset: Dataset, n_splits: int, validation_ratio: float, random_state: int,
     *, quality: CurveQuality) -> list[Fold]

``quality`` is the curve-quality rule the caller asked for. A splitter receives the unfiltered
dataset and decides which pairs to use, normally ``observed_mask(dataset, quality)``; it may
deliberately ignore the rule. Registering a function without a ``quality`` parameter raises.

Register custom splitters with the ``@register`` decorator; validation runs automatically
after each split, with no way to bypass it. A mode's aliases are stored in its registry
metadata, so :func:`canonical` and a registry snapshot see them together with the mode.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Protocol

from drevalpy.errors import InvalidInputError
from drevalpy.registry._registry import Registry
from drevalpy.types import Dataset, Fold

from ._validation import Validation, validate_folds

if TYPE_CHECKING:
    from drevalpy.data.quality import CurveQuality


class Splitter(Protocol):
    """Protocol defining the splitter callable signature."""

    def __call__(
        self,
        dataset: Dataset,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
        *,
        quality: CurveQuality,
    ) -> list[Fold]:
        """Execute the splitter."""
        ...


#: Every registered splitter, wrapped with its validation, under its canonical mode name.
splitter_registry: Registry[Splitter] = Registry("split mode")


def _wrap_with_validation(fn: Splitter, mode: str, validation: Validation) -> Splitter:
    """Wrap a splitter so validation runs and default metadata is injected."""

    @wraps(fn)
    def wrapper(
        dataset: Dataset,
        n_splits: int = 5,
        validation_ratio: float = 0.1,
        random_state: int = 42,
        *,
        quality: CurveQuality,
    ) -> list[Fold]:
        folds = fn(dataset, n_splits, validation_ratio, random_state, quality=quality)
        validate_folds(folds, validation, dataset)
        for i, fold in enumerate(folds):
            fold.metadata.setdefault("split_mode", mode)
            fold.metadata.setdefault("fold_index", i)
            fold.metadata.setdefault("n_splits", n_splits)
            fold.metadata.setdefault("validation_ratio", validation_ratio)
            fold.metadata.setdefault("random_state", random_state)
        return folds

    return wrapper  # type: ignore[return-value]


def _require_quality_parameter(fn: Splitter, mode: str) -> None:
    """Reject a splitter that cannot receive the ``quality`` keyword.

    :param fn: Splitter being registered.
    :param mode: Mode name, for the message.
    :raises InvalidInputError: If *fn* has neither a ``quality`` parameter nor ``**kwargs``.
    """
    parameters = inspect.signature(fn).parameters
    if "quality" not in parameters and not any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        raise InvalidInputError(
            f"Splitter {mode!r} must accept a keyword parameter 'quality: CurveQuality'; "
            "add `*, quality` to its signature and call observed_mask(dataset, quality)."
        )


def _alias_owners() -> dict[str, str]:
    """Return every registered alias with the mode it resolves to.

    :returns: Mapping of alias to canonical mode name.
    """
    return {alias: mode for mode in splitter_registry.names() for alias in splitter_registry.metadata(mode)["aliases"]}


def _check_aliases(mode: str, aliases: tuple[str, ...]) -> None:
    """Reject a mode that shadows an alias, and aliases that shadow a mode or belong to another mode.

    :param mode: Mode name being registered.
    :param aliases: Aliases requested for *mode*.
    :raises ValueError: On any of the three clashes.
    """
    owners = _alias_owners()
    if mode in owners:
        raise ValueError(f"Splitter mode {mode!r} conflicts with an alias of {owners[mode]!r}")
    for alias in aliases:
        if alias == mode or alias in splitter_registry:
            raise ValueError(f"Splitter alias {alias!r} conflicts with the registered mode {alias!r}")
        owner = owners.get(alias)
        if owner is not None and owner != mode:
            raise ValueError(f"Splitter alias {alias!r} already registered for mode {owner!r}")


def register(
    mode: str,
    description: str,
    validation: Validation,
    *,
    replace: bool = False,
    aliases: tuple[str, ...] = (),
) -> Callable[[Splitter], Splitter]:
    """Return a decorator that registers a splitter function under a mode name.

    The splitter is wrapped so that validation runs after every call. Registering a mode
    that is already taken raises ``ValueError`` unless *replace* is true, and so does an
    alias that collides with another mode or alias; a function without a ``quality``
    parameter raises ``InvalidInputError``.

    :param mode: Mode name (e.g. ``"LPO"``, or a custom name).
    :param description: Human-readable description of the splitting approach.
    :param validation: Which leakage constraint to enforce (``"LSO"``, ``"LDO"``, ``"LPO"``,
        ``"LTO"``, ``"LStO"``).
    :param replace: Replace an already-registered mode, with its aliases, instead of raising.
    :param aliases: Alternative names that resolve to *mode*. Results, metadata and
        file names always use *mode*, never the alias the caller typed.
    :returns: Decorator that registers and returns the wrapped function.

    Example::

        @register("MY_LSO", "Custom LSO with fraction", validation="LSO")
        def my_lso(dataset, n_splits=5, validation_ratio=0.1, random_state=42, *, quality): ...
    """

    def decorator(fn: Splitter) -> Splitter:
        _require_quality_parameter(fn, mode)
        _check_aliases(mode, aliases)
        metadata = {
            "description": description,
            "validation": validation,
            "aliases": frozenset(aliases),
        }
        return splitter_registry.register(
            mode, _wrap_with_validation(fn, mode, validation), metadata=metadata, replace=replace
        )

    return decorator


def canonical(mode: str) -> str:
    """Return the canonical mode name for *mode*, resolving an alias.

    Names that are not aliases are returned unchanged, so unknown modes still
    fail later with the usual "Unknown split mode" error.

    :param mode: Mode name or alias.
    :returns: Canonical mode name.
    """
    if mode in splitter_registry:
        return mode
    return _alias_owners().get(mode, mode)


def get(mode: str) -> Splitter:
    """Return the validated splitter for a mode name or alias.

    An unknown *mode* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param mode: Registered mode name or alias.
    :returns: Splitter callable, with validation baked in.
    """
    return splitter_registry.get(canonical(mode))

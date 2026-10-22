"""Comparing and merging ``uns`` dictionaries for :func:`drevalpy.data.concat`.

``uns`` holds free-form metadata: recorded units, the response column, seeds.
Some keys must agree between the inputs for the result to mean anything (a unit,
say); the rest are kept when every input agrees and dropped with a warning
otherwise.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np

from drevalpy.log import get_logger

logger = get_logger(__name__)

_MISSING = object()
_REPR_WIDTH = 60


def same(first: Any, second: Any) -> bool:
    """Whether two ``uns`` values are equal, looking inside dicts, lists and arrays.

    Args:
        first: One value.
        second: Another value.

    Returns:
        ``True`` if they are equal; values that cannot be compared are not equal.
    """
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(same(first[key], second[key]) for key in first)
    if isinstance(first, (np.ndarray, list, tuple)) or isinstance(second, (np.ndarray, list, tuple)):
        try:
            return bool(np.array_equal(np.asarray(first), np.asarray(second)))
        except (TypeError, ValueError):
            return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _short(value: Any) -> str:
    """A bounded ``repr`` for error messages."""
    if value is _MISSING:
        return "not recorded"
    text = repr(value)
    return text if len(text) <= _REPR_WIDTH else text[: _REPR_WIDTH - 3] + "..."


def _agree(values: Sequence[Any]) -> bool:
    """Whether every input recorded the key, with the same value."""
    return values[0] is not _MISSING and all(same(values[0], value) for value in values[1:])


def merge_uns(
    uns_list: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    where: str,
    strict: Callable[[str], bool] = lambda key: False,
    skip: Collection[str] = (),
) -> dict[str, Any]:
    """Merge the ``uns`` of several inputs, keeping only what every input agrees on.

    Args:
        uns_list: One ``uns`` mapping per input.
        names: Dataset name of each input, for messages.
        where: What the ``uns`` belongs to, such as ``"view 'gene_expression'"``.
        strict: Tells whether a key must be equal in every input. A key that is
            not strict is dropped with a warning when the inputs disagree.
        skip: Keys to leave out without a warning.

    Returns:
        A new dict holding copies of the agreed values.

    Raises:
        ValueError: If a strict key differs between inputs, or is recorded by only some.
    """
    keys = [key for key in dict.fromkeys(key for uns in uns_list for key in uns) if key not in skip]
    merged: dict[str, Any] = {}
    dropped: list[str] = []
    for key in keys:
        values = [uns.get(key, _MISSING) for uns in uns_list]
        if _agree(values):
            merged[key] = copy.deepcopy(values[0])
        elif strict(key):
            seen = ", ".join(f"{name}={_short(value)}" for name, value in zip(names, values, strict=True))
            raise ValueError(f"{where}: uns[{key!r}] must be equal in every dataset, got {seen}")
        else:
            dropped.append(key)
    if dropped:
        logger.warning("concat dropped %s uns keys that differ between datasets: %s", where, ", ".join(dropped))
    return merged

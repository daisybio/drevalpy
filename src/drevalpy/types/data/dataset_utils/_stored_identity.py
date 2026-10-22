"""Read back the identity a dataset stores in ``mdata.uns``.

An ``.h5mu`` file returns every list as a NumPy array and every number as a NumPy scalar
(``uns["randomization"]`` comes back as an ``ndarray`` of two strings), so the values a
:class:`~drevalpy.types.data.dataset.Dataset` wrote are not the values it reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def stored_randomization(uns: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return the ``(mode, view)`` pair stored under ``uns["randomization"]``.

    Args:
        uns: The ``uns`` mapping of a loaded MuData object.

    Returns:
        The pair as plain strings, or ``None`` when no two-element value is stored.
    """
    stored = uns.get("randomization")
    if not isinstance(stored, (list, tuple, np.ndarray)) or len(stored) != 2:
        return None
    return str(stored[0]), str(stored[1])


def plain_value(value: Any) -> Any:
    """Convert NumPy arrays and scalars inside *value* to JSON-compatible Python objects.

    Args:
        value: A stored value, possibly a nested mapping or list.

    Returns:
        *value* with every ``ndarray`` as a list and every NumPy scalar as a Python scalar.
    """
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return plain_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value

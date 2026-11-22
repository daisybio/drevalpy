"""Reading what a dataset stores in an AnnData or in ``mdata.uns``.

Matrices are densified and layer names are the real ones. An ``.h5mu`` file returns every list as a NumPy array
and every number as a NumPy scalar (``uns["randomization"]`` comes back as an ``ndarray`` of two strings), so the
values a :class:`~drevalpy.types.data.dataset.Dataset` wrote are not the values it reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import anndata as ad


def to_dense(x: Any) -> Any:
    """Return *x* densified if it is a SciPy sparse matrix, otherwise unchanged.

    AnnData stores ``X``, layers and ``varm`` entries either as dense arrays or as
    SciPy sparse matrices depending on how the ``.h5mu`` was written, so callers
    cannot know which they hold.

    :param x: Matrix-like object, dense or sparse.
    :returns: ``x.toarray()`` when *x* is sparse, else *x* itself.
    """
    from scipy.sparse import issparse

    return x.toarray() if issparse(x) else x


def layer_names(adata: ad.AnnData) -> list[str]:
    """Return the layer names of *adata*, in insertion order.

    anndata 0.13 yields a spurious ``None`` from ``layers.keys()``, even for a freshly
    constructed AnnData with a single layer, so only the ``str`` keys are real names.

    :param adata: AnnData whose layers to list.
    :returns: The layer names.
    """
    return [name for name in adata.layers.keys() if isinstance(name, str)]


def stored_randomization(uns: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return the ``(mode, view)`` pair stored under ``uns["randomization"]``.

    :param uns: The ``uns`` mapping of a loaded MuData object.
    :returns: The pair as plain strings, or ``None`` when no two-element value is stored.
    """
    stored = uns.get("randomization")
    if not isinstance(stored, (list, tuple, np.ndarray)) or len(stored) != 2:
        return None
    return str(stored[0]), str(stored[1])


def plain_value(value: Any) -> Any:
    """Convert NumPy arrays and scalars inside *value* to JSON-compatible Python objects.

    :param value: A stored value, possibly a nested mapping or list.
    :returns: *value* with every ``ndarray`` as a list and every NumPy scalar as a Python scalar.
    """
    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return plain_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value

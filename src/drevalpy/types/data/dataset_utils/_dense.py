"""Densification helper shared by every matrix read on the dataset hot path."""

from __future__ import annotations

from typing import Any


def to_dense(x: Any) -> Any:
    """Return *x* densified if it is a SciPy sparse matrix, otherwise unchanged.

    AnnData stores ``X``, layers and ``varm`` entries either as dense arrays or as
    SciPy sparse matrices depending on how the ``.h5mu`` was written, so callers
    cannot know which they hold. Only sparse containers expose ``toarray``.

    Args:
        x: Matrix-like object, dense or sparse.

    Returns:
        A dense array-like: ``x.toarray()`` when *x* is sparse, else *x* itself.
    """
    to_array = getattr(x, "toarray", None)
    return to_array() if callable(to_array) else x

"""Densification helper shared by every matrix read on the dataset hot path."""

from __future__ import annotations

from typing import Any


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

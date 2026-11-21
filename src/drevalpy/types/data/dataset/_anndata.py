"""Reading an AnnData the same way everywhere: densified matrices and real layer names."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

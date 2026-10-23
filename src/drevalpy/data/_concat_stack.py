"""Row stacking of sample views and response-shaped modalities for :func:`drevalpy.data.concat`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger
from drevalpy.types.data.dataset_utils._dense import to_dense

from ._concat_tables import listed, prefixed, stack_obs, union_var
from ._concat_uns import merge_uns

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd

logger = get_logger(__name__)


def _is_unit_key(key: str) -> bool:
    """Whether a view's ``uns`` key records a unit, such as ``expression_unit``."""
    return key.endswith(("_unit", "_units"))


def _layer_names(adata: ad.AnnData) -> list[str]:
    """Real layer names; anndata can yield a spurious ``None`` key."""
    return [name for name in adata.layers.keys() if isinstance(name, str)]


def _prefixed_obs(adatas: Sequence[ad.AnnData], names: Sequence[str]) -> list[pd.DataFrame]:
    """Each input's ``obs`` with ``<study>::`` prefixed to its row names."""
    pairs = zip(adatas, names, strict=True)
    return [adata.obs.set_axis(prefixed(name, adata.obs_names), axis=0) for adata, name in pairs]


def _vstack(blocks: Sequence[Any]) -> Any:
    """Stack matrices row-wise, sparse if any of them is."""
    from scipy import sparse

    if any(sparse.issparse(block) for block in blocks):
        return sparse.vstack(blocks, format="csr")
    return np.vstack(blocks)


def _feature_mismatch(
    view: str, names: Sequence[str], index: int, first: pd.Index, other: pd.Index
) -> InvalidInputError:
    """Build the error for a view whose feature names differ from the first input's."""
    missing = list(first.difference(other, sort=False))
    extra = list(other.difference(first, sort=False))
    if missing or extra:
        detail = f"{len(missing)} features of {names[0]} missing in {names[index]} (e.g. {listed(missing)})"
        if extra:
            detail += f"; {len(extra)} only in {names[index]} (e.g. {listed(extra)})"
    else:
        detail = "the same features in a different order"
    return InvalidInputError(f"View '{view}' has different feature names in {names[0]} and {names[index]}: {detail}")


def _view_columns(
    view: str, adatas: Sequence[ad.AnnData], names: Sequence[str], join: Literal["exact", "inner"]
) -> pd.Index:
    """Return the feature names the stacked view keeps."""
    first = adatas[0].var_names
    if join == "exact":
        for index, adata in enumerate(adatas[1:], start=1):
            if not first.equals(adata.var_names):
                raise _feature_mismatch(view, names, index, first, adata.var_names)
        return first
    shared = first
    for adata in adatas[1:]:
        shared = shared[shared.isin(adata.var_names)]
    if len(shared) == 0:
        raise InvalidInputError(f"View '{view}' has no feature names shared by {', '.join(names)}")
    return shared


def _select_columns(adata: ad.AnnData, matrix: Any, columns: pd.Index) -> Any:
    """Reorder or subset the feature axis of *matrix* to *columns*."""
    if adata.var_names.equals(columns):
        return matrix
    return matrix[:, adata.var_names.get_indexer(columns)]


def _stack_layers(view: str, adatas: Sequence[ad.AnnData], columns: pd.Index) -> dict[str, Any]:
    """Stack the layers every input has; warn about the others."""
    per_input = [set(_layer_names(adata)) for adata in adatas]
    common = [name for name in _layer_names(adatas[0]) if all(name in names for names in per_input)]
    lost = sorted(set().union(*per_input) - set(common))
    if lost:
        logger.warning("concat dropped layers of view '%s' that are not in every dataset: %s", view, ", ".join(lost))
    return {name: _vstack([_select_columns(adata, adata.layers[name], columns) for adata in adatas]) for name in common}


def stack_view(
    view: str, adatas: Sequence[ad.AnnData], names: Sequence[str], join: Literal["exact", "inner"]
) -> ad.AnnData:
    """Stack one sample view row-wise over all inputs.

    :param view: The modality key, for messages.
    :param adatas: The view of each input, in input order.
    :param names: Dataset name of each input; it prefixes the row IDs.
    :param join: ``"exact"`` needs identical feature names; ``"inner"`` keeps the shared ones
        in the first input's order.
    :returns: A new AnnData with ``<study>::<sample_id>`` row names. Recorded units must be
        equal in every input under both joins.
    :raises InvalidInputError: If feature names differ under ``"exact"``, none are shared under
        ``"inner"``, or recorded units differ.
    """
    import anndata as ad

    columns = _view_columns(view, adatas, names, join)
    uns = merge_uns([adata.uns for adata in adatas], names=names, where=f"view '{view}'", strict=_is_unit_key)
    rows = _prefixed_obs(adatas, names)
    matrix = _vstack([_select_columns(adata, adata.X, columns) for adata in adatas])
    var = adatas[0].var.loc[columns].copy()
    return ad.AnnData(X=matrix, obs=stack_obs(rows), var=var, layers=_stack_layers(view, adatas, columns), uns=uns)


def _fill(blocks: Sequence[np.ndarray], offsets: Sequence[np.ndarray], n_columns: int) -> np.ndarray:
    """Place each block at its rows (stacked in order) and the columns in *offsets*; NaN elsewhere."""
    dtype = np.result_type(np.float32, *(block.dtype for block in blocks))
    out = np.full((sum(len(block) for block in blocks), n_columns), np.nan, dtype=dtype)
    start = 0
    for block, columns in zip(blocks, offsets, strict=True):
        out[start : start + len(block), columns] = block
        start += len(block)
    return out


def stack_response(
    adatas: Sequence[ad.AnnData],
    names: Sequence[str],
    *,
    where: str = "response",
    strict_uns: Callable[[str], bool] = lambda key: True,
) -> ad.AnnData:
    """Stack a response-shaped modality: rows stacked, drug columns and layers unioned.

    :param adatas: The modality of each input, in input order.
    :param names: Dataset name of each input; it prefixes the row IDs.
    :param where: The modality key, for messages.
    :param strict_uns: Tells which ``uns`` keys must be equal in every input; the others
        are kept when equal and dropped with a warning otherwise.
    :returns: A new AnnData over ``<study>::<sample_id>`` rows and the union of the drug IDs
        in first-seen order. ``X`` and every layer are NaN where an input lacks the drug
        or the layer.
    :raises InvalidInputError: If a drug has different ``canonical_smiles`` in different inputs,
        or a strict ``uns`` key differs.
    """
    import anndata as ad

    var = union_var([adata.var for adata in adatas])
    offsets = [var.index.get_indexer(adata.var_names) for adata in adatas]
    rows = _prefixed_obs(adatas, names)
    layer_names = list(dict.fromkeys(name for adata in adatas for name in _layer_names(adata)))
    layers = {}
    for layer in layer_names:
        blocks = [
            to_dense(adata.layers[layer]) if layer in adata.layers else np.full(adata.shape, np.nan, dtype=np.float32)
            for adata in adatas
        ]
        layers[layer] = _fill(blocks, offsets, len(var))
    uns = merge_uns([adata.uns for adata in adatas], names=names, where=where, strict=strict_uns)
    x = _fill([to_dense(adata.X) for adata in adatas], offsets, len(var))
    return ad.AnnData(X=x, obs=stack_obs(rows), var=var, layers=layers, uns=uns)

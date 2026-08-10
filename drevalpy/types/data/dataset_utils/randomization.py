"""Randomization utilities for Dataset views."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mudata as md
import numpy as np

from drevalpy.log import get_logger

if TYPE_CHECKING:
    from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _randomize_matrix(data: np.ndarray, rng: np.random.Generator, randomization_type: str) -> np.ndarray:
    """Apply randomization to a 2-D feature matrix."""
    if randomization_type == "permutation":
        perm = rng.permutation(data.shape[0])
        return data[perm]
    return np.array(
        [rng.normal(row.mean(), max(row.std(), 1e-8), row.shape) for row in data],
        dtype=np.float32,
    )


def _randomize_single_view(
    dataset: Any,
    view: str,
    new_mods: dict[str, md.AnnData],
    new_uns: dict[str, Any],
    rng: np.random.Generator,
    randomization_type: str,
) -> None:
    """Randomize a single view in-place within new_mods/new_uns."""
    import anndata

    if view in new_mods and view != "response":
        adata = new_mods[view]
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = _randomize_matrix(np.asarray(x, dtype=np.float32), rng, randomization_type)
        new_mods[view] = anndata.AnnData(X=x, obs=adata.obs.copy(), var=adata.var.copy())
    elif "response" in new_mods and view in (new_mods["response"].varm or {}):
        resp = new_mods["response"]
        varm_data = np.asarray(resp.varm[view], dtype=np.float32)
        resp.varm[view] = _randomize_matrix(varm_data, rng, randomization_type)
    elif "response" in new_mods and view in (new_mods["response"].obsm or {}):
        resp = new_mods["response"]
        obsm_data = np.asarray(resp.obsm[view], dtype=np.float32)
        resp.obsm[view] = _randomize_matrix(obsm_data, rng, randomization_type)
    elif view in new_uns:
        data = new_uns[view]
        if isinstance(data, dict):
            keys = list(data.keys())
            shuffled_keys = rng.permutation(keys).tolist()
            new_uns[view] = dict(zip(shuffled_keys, data.values(), strict=True))
        else:
            logger.warning("Cannot randomize uns key '%s' (not a dict). Skipping.", view)
    else:
        logger.warning("View '%s' not found in any storage location. Skipping randomization.", view)


def with_randomized_views(
    dataset: Dataset,
    views: list[str],
    randomization_type: str = "permutation",
    random_state: int | None = None,
    *,
    randomization: tuple[str, str] | None = None,
) -> Dataset:
    """Return a copy of a Dataset with specified views randomized.

    For cell-line views (modalities or obsm keys), rows are permuted across
    cell lines. For drug views (varm keys), rows are permuted across drugs.
    For uns dict keys, values are reassigned to shuffled keys.

    Args:
        dataset: Source Dataset to copy and randomize.
        views: View names to randomize.
        randomization_type: "permutation" shuffles rows; "invariant" replaces
            each row with a random sample matching its mean and std.
        random_state: Seed for reproducibility.
        randomization: Optional (mode, view) tuple to attach to the new dataset.

    Returns:
        A new Dataset with the specified views randomized.
    """
    import copy

    from drevalpy.types.data.dataset import Dataset as DatasetCls

    if randomization_type not in ("permutation", "invariant"):
        raise ValueError(f"Unknown randomization_type {randomization_type!r}. Use 'permutation' or 'invariant'.")

    rng = np.random.default_rng(random_state)

    new_mods: dict[str, md.AnnData] = {}
    for mod_name, mod_adata in dataset._mdata.mod.items():
        new_mods[mod_name] = mod_adata.copy()

    new_uns: dict[str, Any] = {
        key: copy.deepcopy(val) if isinstance(val, dict) else val for key, val in dataset._mdata.uns.items()
    }

    for view in views:
        _randomize_single_view(dataset, view, new_mods, new_uns, rng, randomization_type)

    md.set_options(pull_on_update=False)
    new_mdata = md.MuData(new_mods)
    new_mdata.obs = dataset._mdata.obs.copy()
    for key, val in new_uns.items():
        new_mdata.uns[key] = val
    return DatasetCls(new_mdata, name=dataset._name, randomization=randomization)

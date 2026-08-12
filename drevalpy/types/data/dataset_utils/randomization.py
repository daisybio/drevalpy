"""Mixin providing view randomization for Dataset."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import mudata as md
import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.modalities import backing_modality

from ._dense import to_dense

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


def _degree_preserving_rewire(
    edge_index: np.ndarray, rng: np.random.Generator, n_swaps: int | None = None
) -> np.ndarray:
    """Rewire edges while preserving each node's degree.

    Uses pairwise edge swaps: pick two edges (u,v) and (x,y), replace with
    (u,y) and (x,v) if neither already exists (avoiding self-loops and
    multi-edges).

    Args:
        edge_index: (2, E) array of edge endpoints.
        rng: Numpy random generator.
        n_swaps: Number of swap attempts. Defaults to 10 * num_edges.

    Returns:
        Rewired (2, E) edge_index with identical degree sequence.
    """
    edges = edge_index.T.copy()
    num_edges = edges.shape[0]
    if num_edges < 2:
        return edge_index.copy()

    if n_swaps is None:
        n_swaps = 10 * num_edges

    edge_set: set[tuple[int, int]] = {(int(edges[i, 0]), int(edges[i, 1])) for i in range(num_edges)}

    for _ in range(n_swaps):
        i, j = rng.integers(0, num_edges, size=2)
        if i == j:
            continue
        u, v = int(edges[i, 0]), int(edges[i, 1])
        x, y = int(edges[j, 0]), int(edges[j, 1])

        if u == y or x == v:
            continue
        if (u, y) in edge_set or (x, v) in edge_set:
            continue

        edge_set.discard((u, v))
        edge_set.discard((x, y))
        edge_set.add((u, y))
        edge_set.add((x, v))
        edges[i] = [u, y]
        edges[j] = [x, v]

    return edges.T


def _randomize_graph(graph: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Apply invariant randomization to a single drug graph.

    Preserves degree distribution (edge_index), and replaces node/edge
    features with Gaussian samples matching per-row mean and std.
    """
    result: dict[str, np.ndarray] = {}
    for key, val in graph.items():
        arr = np.asarray(val)
        if key == "edge_index":
            result[key] = _degree_preserving_rewire(arr, rng)
        elif arr.ndim == 2:
            result[key] = _randomize_matrix(arr, rng, "invariant")
        else:
            result[key] = arr.copy()
    return result


def _is_graph_dict(data: dict[str, Any]) -> bool:
    """Check if a dict-of-dicts looks like a collection of graph dicts.

    Heuristic: at least one value is itself a dict containing an "edge_index" key.
    """
    return any(isinstance(v, dict) and "edge_index" in v for v in data.values())


def _randomize_uns_view(
    data: Any,
    view: str,
    new_uns: dict[str, Any],
    rng: np.random.Generator,
    randomization_type: str,
) -> None:
    """Randomize a view stored in uns (dict of dicts or plain dict)."""
    if not isinstance(data, dict):
        logger.warning("Cannot randomize uns key '%s' (not a dict). Skipping.", view)
        return

    if randomization_type == "invariant" and _is_graph_dict(data):
        new_uns[view] = {key: _randomize_graph(val, rng) if isinstance(val, dict) else val for key, val in data.items()}
    else:
        keys = list(data.keys())
        shuffled_keys = rng.permutation(keys).tolist()
        new_uns[view] = dict(zip(shuffled_keys, data.values(), strict=True))


def _randomize_single_view(
    dataset: Any,
    view: str,
    new_mods: dict[str, md.AnnData],
    new_uns: dict[str, Any],
    rng: np.random.Generator,
    randomization_type: str,
) -> None:
    """Randomize a single view in-place within new_mods/new_uns.

    *view* is a public name, so the omics modalities are looked up through the
    accessor map; varm, obsm and uns keys are not omics and stay verbatim.
    """
    import anndata

    modality = backing_modality(view, new_mods)
    if modality is not None and modality != "response":
        adata = new_mods[modality]
        x = _randomize_matrix(np.asarray(to_dense(adata.X), dtype=np.float32), rng, randomization_type)
        new_mods[modality] = anndata.AnnData(X=x, obs=adata.obs.copy(), var=adata.var.copy())
    elif "response" in new_mods and view in (new_mods["response"].varm or {}):
        resp = new_mods["response"]
        varm_data = np.asarray(resp.varm[view], dtype=np.float32)
        resp.varm[view] = _randomize_matrix(varm_data, rng, randomization_type)
    elif "response" in new_mods and view in (new_mods["response"].obsm or {}):
        resp = new_mods["response"]
        obsm_data = np.asarray(resp.obsm[view], dtype=np.float32)
        resp.obsm[view] = _randomize_matrix(obsm_data, rng, randomization_type)
    elif view in new_uns:
        _randomize_uns_view(new_uns[view], view, new_uns, rng, randomization_type)
    else:
        logger.warning("View '%s' not found in any storage location. Skipping randomization.", view)


class RandomizationMixin:
    """Mixin that provides view randomization for Dataset.

    Expects ``self._mdata`` to be a MuData object and ``self._name`` to be the dataset name.
    """

    _mdata: md.MuData
    _name: str

    def with_randomized_views(
        self,
        views: list[str],
        randomization_type: str = "permutation",
        random_state: int | None = None,
        *,
        randomization: tuple[str, str] | None = None,
    ) -> Dataset:
        """Return a copy of this Dataset with specified views randomized.

        For cell-line views (modalities or obsm keys), rows are permuted across
        cell lines. For drug views (varm keys), rows are permuted across drugs.
        For uns dict keys, values are reassigned to shuffled keys.

        Args:
            views: View names to randomize.
            randomization_type: "permutation" shuffles rows; "invariant" replaces
                each row with a random sample matching its mean and std.
            random_state: Seed for reproducibility.
            randomization: Optional (mode, view) tuple to attach to the new dataset.

        Returns:
            A new Dataset with the specified views randomized.

        Raises:
            ValueError: If randomization_type is not recognized.
            KeyError: If a view is not found in any storage location.
        """
        from drevalpy.types.data.dataset import Dataset as DatasetCls

        if randomization_type not in ("permutation", "invariant"):
            raise ValueError(f"Unknown randomization_type {randomization_type!r}. Use 'permutation' or 'invariant'.")

        rng = np.random.default_rng(random_state)

        new_mods: dict[str, md.AnnData] = {}
        for mod_name, mod_adata in self._mdata.mod.items():
            new_mods[mod_name] = mod_adata.copy()

        new_uns: dict[str, Any] = {
            key: copy.deepcopy(val) if isinstance(val, dict) else val for key, val in self._mdata.uns.items()
        }

        for view in views:
            _randomize_single_view(self, view, new_mods, new_uns, rng, randomization_type)

        md.set_options(pull_on_update=False)
        new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.copy()
        for key, val in new_uns.items():
            new_mdata.uns[key] = val
        return DatasetCls(new_mdata, name=self._name, randomization=randomization)

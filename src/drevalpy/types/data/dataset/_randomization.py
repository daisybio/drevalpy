"""Mixin providing view randomization for Dataset."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.views import find_view
from drevalpy.types.enums.randomization import RandomizationType

from ._dense import to_dense

if TYPE_CHECKING:
    import mudata as md

    from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _randomize_matrix(data: np.ndarray, rng: np.random.Generator, randomization_type: RandomizationType) -> np.ndarray:
    """Apply randomization to a 2-D feature matrix."""
    if randomization_type == RandomizationType.PERMUTATION:
        return rng.permutation(data, axis=0)
    spread = np.maximum(data.std(axis=1, keepdims=True), 1e-8)
    return rng.normal(data.mean(axis=1, keepdims=True), spread, size=data.shape).astype(np.float32)


def _degree_preserving_rewire(edge_index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rewire the bonds of a graph while preserving each node's degree.

    Drug graphs store every bond in both directions, so the swaps run on the undirected graph
    and both directions are emitted again.

    :param edge_index: (2, E) array of edge endpoints, each bond in both directions.
    :param rng: Numpy random generator.
    :returns: Rewired (2, E) edge_index, or *edge_index* itself when the graph has fewer than two bonds or no
        swap is possible.
    """
    import networkx as nx

    bonds = np.unique(np.sort(edge_index.T, axis=1), axis=0)
    if len(bonds) < 2:
        return edge_index
    graph = nx.Graph(bonds.tolist())
    try:
        nx.double_edge_swap(graph, nswap=10 * len(bonds), max_tries=1000 * len(bonds), seed=rng)
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        return edge_index
    swapped = np.array(graph.edges, dtype=edge_index.dtype)
    return np.column_stack([swapped, swapped[:, ::-1]]).reshape(-1, 2).T


def _randomize_graph(graph: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Apply invariant randomization to a single drug graph.

    Preserves degree distribution (edge_index), and replaces node/edge
    features with Gaussian draws matching per-row mean and std.
    """
    result: dict[str, np.ndarray] = {}
    for key, val in graph.items():
        arr = np.asarray(val)
        if key == "edge_index":
            result[key] = _degree_preserving_rewire(arr, rng)
        elif arr.ndim == 2:
            result[key] = _randomize_matrix(arr, rng, RandomizationType.INVARIANT)
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
    randomization_type: RandomizationType,
) -> None:
    """Randomize a view stored in uns (dict of dicts or plain dict)."""
    if not isinstance(data, dict):
        logger.warning("Cannot randomize uns key '%s' (not a dict). Skipping.", view)
        return

    if randomization_type == RandomizationType.INVARIANT and _is_graph_dict(data):
        new_uns[view] = {key: _randomize_graph(val, rng) if isinstance(val, dict) else val for key, val in data.items()}
    else:
        keys = list(data.keys())
        shuffled_keys = rng.permutation(keys).tolist()
        new_uns[view] = dict(zip(shuffled_keys, data.values(), strict=True))


def _randomize_single_view(
    view: str,
    new_mods: dict[str, md.AnnData],
    new_uns: dict[str, Any],
    rng: np.random.Generator,
    randomization_type: RandomizationType,
) -> None:
    """Randomize a single view in-place within new_mods/new_uns.

    *view* is a public name, so a sample view is looked up through its stored
    modality name; varm, obsm and uns keys stay verbatim.
    """
    stored = find_view(view, new_mods)
    if stored is not None and stored != "response":
        adata = new_mods[stored]
        adata.X = _randomize_matrix(np.asarray(to_dense(adata.X), dtype=np.float32), rng, randomization_type)
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
        raise KeyError(f"View '{view}' not found in any storage location of the dataset.")


class RandomizationMixin:
    """Mixin that provides view randomization for Dataset.

    Expects ``self._mdata`` to be a MuData object and ``self.name`` to be the dataset name.
    """

    _mdata: md.MuData
    name: str

    def with_randomized_views(
        self,
        views: list[str],
        randomization_type: RandomizationType = RandomizationType.PERMUTATION,
        random_state: int | None = None,
        *,
        randomization: tuple[str, str] | None = None,
    ) -> Dataset:
        """Return a copy of this Dataset with specified views randomized.

        For sample views (modalities or obsm keys), rows are permuted across
        samples. For drug views (varm keys), rows are permuted across drugs.
        For uns dict keys, values are reassigned to shuffled keys.

        Raises ``KeyError`` if a view is not found in any storage location.

        :param views: View names to randomize.
        :param randomization_type: ``permutation`` shuffles rows; ``invariant`` replaces
            each row with a random draw matching its mean and std. A value that is
            not a :class:`RandomizationType` is rejected.
        :param random_state: Seed for reproducibility.
        :param randomization: Optional (mode, view) tuple to attach to the new dataset.
        :returns: A new Dataset with the specified views randomized.
        """
        import mudata as md

        from drevalpy.types.data.dataset import Dataset as DatasetCls

        randomization_type = RandomizationType(randomization_type)
        rng = np.random.default_rng(random_state)

        new_mods: dict[str, md.AnnData] = {}
        for mod_name, mod_adata in self._mdata.mod.items():
            new_mods[mod_name] = mod_adata.copy()

        new_uns: dict[str, Any] = {
            key: copy.deepcopy(val) if isinstance(val, dict) else val for key, val in self._mdata.uns.items()
        }

        for view in views:
            _randomize_single_view(view, new_mods, new_uns, rng, randomization_type)

        with md.set_options(pull_on_update=False):
            new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.copy()
        for key, val in new_uns.items():
            new_mdata.uns[key] = val
        return DatasetCls(new_mdata, name=self.name, randomization=randomization)

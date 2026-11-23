"""Randomized featurizer output, behind the randomization tests of :doc:`/reference/split_modes`.

A model trained on a dataset with a
:attr:`~drevalpy.types.data.dataset.Dataset.view_randomization` wraps each leaf featurizer that
reads one of the randomized views in a :class:`RandomizedFeaturizer`; inside a concat, only the
children that read such a view are wrapped. The randomization applies to what the featurizer
emits, at its own settings and in every format, and never to the stored data. The featurizer is
fitted on the real input: only ``transform`` is randomized.

- ``permutation`` gives every entity the output of a donor entity. The donors are one seeded
  permutation of the entities in the response matrix that the featurizer can use, so an entity
  has the same donor in every ``transform`` of a fit and of its predictions, and an unusable entity
  keeps its own unusable output. Leaves that read the same views get the same permutation.
- ``invariant`` replaces every row of an entity's output (one row of a matrix block, every atom of a
  ragged block or a graph) with a Gaussian draw of that row's mean and standard deviation (at least
  ``1e-8``), and rewires the bonds of a graph keeping the degree of every atom. Each entity's draws
  are seeded from its ID, so they too are the same in every ``transform``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from drevalpy._utils.seed import keyed_seed
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.feature_format import FeatureFormat
from drevalpy.types.enums import RandomizationType

if TYPE_CHECKING:
    from drevalpy.types.data.dataset._randomization import ViewRandomization
    from drevalpy.types.data.feature_source import FeatureSource


class RandomizedFeaturizer(Featurizer):
    """Fit a featurizer on the real input and randomize what it emits per entity."""

    def __init__(self, featurizer: Featurizer, randomization: ViewRandomization, views: tuple[str, ...]) -> None:
        """Wrap *featurizer*, which reads *views*.

        :param featurizer: The built featurizer to randomize.
        :param randomization: The randomization of the dataset the model is trained on.
        :param views: The views *featurizer* reads; they key the seed of its permutation.
        """
        self.featurizer = featurizer
        self.randomization_type = randomization.randomization_type
        self.seed = keyed_seed(randomization.seed, ",".join(sorted(views)))
        # The class-level declarations of the wrapped featurizer, which callers read on the instance.
        self.side = featurizer.side  # ty: ignore[invalid-attribute-access]
        self.nan_threshold = featurizer.nan_threshold  # ty: ignore[invalid-attribute-access]
        self.requires_early_stopping = featurizer.requires_early_stopping  # ty: ignore[invalid-attribute-access]

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return the usable entities of the wrapped featurizer; permutation keeps them usable.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return self.featurizer.valid_mask(source, ids)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit the wrapped featurizer on the real input.

        :param source: Feature source of the featurizer's side.
        :param train_ids: Usable training entity IDs.
        :param train_pair_ids: Entity ID of every usable training pair.
        :param es_pair_ids: Entity ID of every usable early-stopping pair.
        :returns: This featurizer, fitted.
        """
        self.featurizer.fit(source, train_ids, train_pair_ids=train_pair_ids, es_pair_ids=es_pair_ids)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        """Return the randomized output of the wrapped featurizer for *ids*.

        :param source: Feature source of the featurizer's side.
        :param ids: Usable entity IDs, in output order.
        :returns: Blocks with one row per ID; blocks that are not entity-aligned are unchanged.
        """
        if self.randomization_type == RandomizationType.PERMUTATION:
            return FeaturizerOutput(ids, self.featurizer.transform(source, self._donors(source, ids)).blocks)
        output = self.featurizer.transform(source, ids)
        rngs = [np.random.default_rng(keyed_seed(self.seed, str(entity))) for entity in ids]
        blocks = {
            name: _resampled_block(block, rngs) if block.entity_aligned else block
            for name, block in output.blocks.items()
        }
        return FeaturizerOutput(ids, blocks)

    def _donors(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return the entity whose output each of *ids* gets under the permutation.

        :param source: Feature source of the featurizer's side.
        :param ids: Entity IDs.
        :returns: One donor ID per ID; an entity outside the permuted ones is its own donor.
        """
        entities = source.entity_ids()
        usable = np.sort(entities[self.featurizer.valid_mask(source, entities)])
        donor_of = dict(
            zip(usable.tolist(), np.random.default_rng(self.seed).permutation(usable).tolist(), strict=True)
        )
        donors = [donor_of.get(entity, entity) for entity in ids.tolist()]
        return np.asarray(donors, dtype=np.result_type(ids.dtype, usable.dtype))


def _resampled_block(block: FeatureBlock, rngs: list[np.random.Generator]) -> FeatureBlock:
    """Return *block* with the output of every entity drawn anew, entity ``i`` from ``rngs[i]``.

    :param block: Entity-aligned block.
    :param rngs: One generator per entity.
    :returns: The block with the same format and names.
    """
    numeric = block.format == FeatureFormat.NUMERIC_MATRIX
    values = np.empty(block.values.shape, np.float32) if numeric else np.empty(len(block.values), object)
    for i, (value, rng) in enumerate(zip(block.values, rngs, strict=True)):
        if value is not None:
            values[i] = _resampled_graph(value, rng) if block.format == FeatureFormat.GRAPH else _resampled_rows(value, rng)
    return FeatureBlock(values, block.format, block.feature_names, block.annotations)


def _resampled_rows(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw every row of *values* from a Gaussian with that row's mean and standard deviation.

    :param values: 1-D row or 2-D array of rows.
    :param rng: Generator of the draws.
    :returns: float32 array of the same shape.
    """
    rows = np.atleast_2d(values)
    spread = np.maximum(rows.std(axis=1, keepdims=True), 1e-8)
    drawn = rng.normal(rows.mean(axis=1, keepdims=True), spread, size=rows.shape)
    return drawn.astype(np.float32).reshape(np.shape(values))


def _resampled_graph(graph: Any, rng: np.random.Generator) -> Any:
    """Rewire the bonds of a PyG graph, keeping every atom's degree, and draw its node and edge features anew.

    :param graph: ``torch_geometric`` ``Data`` with ``edge_index`` and 2-D float features such as ``x``.
    :param rng: Generator of the draws.
    :returns: A new graph; *graph* is not changed.
    """
    import torch

    result = graph.clone()
    for key, value in graph:
        if key == "edge_index":
            result[key] = torch.as_tensor(_degree_preserving_rewire(value.numpy(), rng))
        elif torch.is_tensor(value) and value.dim() == 2 and value.is_floating_point():
            result[key] = torch.as_tensor(_resampled_rows(value.numpy(), rng), dtype=value.dtype)
    return result


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

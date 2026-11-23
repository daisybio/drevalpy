"""``RandomizedFeaturizer``: per-entity permutation and resampling of a featurizer's output."""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.featurizers._randomized import (
    RandomizedFeaturizer,
    _degree_preserving_rewire,
    _resampled_rows,
)
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.models._feature_preparation import prepare_features
from drevalpy.types.data.dataset._randomization import ViewRandomization
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.feature_format import FeatureFormat
from drevalpy.types.data.feature_source import FeatureSource, Side
from drevalpy.types.enums import RandomizationType
from tests.components.featurizers._helpers import mock_source
from tests.conftest import MockFeatureSource

_N_FEATURES = 200
_IDS = np.array([f"s{i}" for i in range(8)])


def _source() -> MockFeatureSource:
    """Eight samples with distinct ``expr`` rows; ``s3`` has an all-NaN row and is unusable."""
    rows = np.random.default_rng(0).normal(size=(len(_IDS), _N_FEATURES)) + np.arange(len(_IDS))[:, None]
    rows[3] = np.nan
    return mock_source({"expr": (rows, [f"g{i}" for i in range(_N_FEATURES)])})


class _Rows(Featurizer):
    """Emits the ``expr`` row of every entity, every row twice as a ragged matrix, and a global block."""

    side: ClassVar[Side] = Side.SAMPLE
    input_views: ClassVar[tuple[str, ...]] = ("expr",)

    def __init__(self) -> None:
        self.fit_ids: np.ndarray | None = None

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        self.fit_ids = train_ids
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        matrix = source.view("expr", ids)
        ragged = np.empty(len(ids), dtype=object)
        ragged[:] = [np.stack([row, row]) for row in matrix]
        return FeaturizerOutput(
            ids,
            {
                "expr": FeatureBlock(matrix),
                "ragged": FeatureBlock(ragged, FeatureFormat.RAGGED_SEQUENCE),
                "global": FeatureBlock(np.arange(3.0), entity_aligned=False),
            },
        )


def _randomized(
    randomization_type: RandomizationType, seed: int = 0, views: tuple[str, ...] = ("expr",)
) -> RandomizedFeaturizer:
    return RandomizedFeaturizer(_Rows(), ViewRandomization(frozenset(views), randomization_type, seed), views)


def _row_owner(matrix: np.ndarray, real: np.ndarray) -> list[int]:
    """Return, for every row of *matrix*, the index of the equal row of *real*."""
    return [int(np.flatnonzero((real == row).all(axis=1))[0]) for row in matrix]


class TestPermutation:
    def test_every_usable_entity_gets_the_output_of_another_and_an_unusable_one_stays_unusable(self):
        source = _source()
        real = source.view("expr", _IDS)

        features = prepare_features(_randomized(RandomizationType.PERMUTATION), source, _IDS)

        usable = [i for i in range(len(_IDS)) if i != 3]
        owners = _row_owner(features.matrix[usable], real)
        assert sorted(owners) == usable
        assert owners != usable
        assert np.isnan(features.matrix[3]).all()

    def test_an_entity_has_the_same_donor_in_every_transform(self):
        source = _source()
        featurizer = _randomized(RandomizationType.PERMUTATION)
        usable = _IDS[_IDS != "s3"]

        whole = featurizer.transform(source, usable)
        parts = [featurizer.transform(source, usable[:3]), featurizer.transform(source, usable[3:][::-1])]

        np.testing.assert_array_equal(whole.matrix, np.vstack([parts[0].matrix, parts[1].matrix[::-1]]))
        np.testing.assert_array_equal(whole.entity_ids, usable)

    def test_every_entity_aligned_block_follows_the_donor_and_a_global_block_is_kept(self):
        output = _randomized(RandomizationType.PERMUTATION).transform(_source(), _IDS[:3])

        for row, ragged in zip(output.blocks["expr"].values, output.blocks["ragged"].values, strict=True):
            np.testing.assert_array_equal(ragged, np.stack([row, row]))
        np.testing.assert_array_equal(output.blocks["global"].values, np.arange(3.0))

    def test_the_seed_and_the_views_key_the_permutation(self):
        source = _source()

        def matrix(**kwargs) -> np.ndarray:
            return _randomized(RandomizationType.PERMUTATION, **kwargs).transform(source, _IDS[:3]).matrix

        np.testing.assert_array_equal(matrix(seed=1), matrix(seed=1))
        assert not np.array_equal(matrix(seed=1), matrix(seed=2))
        assert not np.array_equal(matrix(seed=1), matrix(seed=1, views=("other",)))

    def test_fit_sees_the_real_entities(self):
        featurizer = _randomized(RandomizationType.PERMUTATION)

        featurizer.fit(_source(), _IDS[:4])

        assert isinstance(featurizer.featurizer, _Rows)
        np.testing.assert_array_equal(featurizer.featurizer.fit_ids, _IDS[:4])


class TestInvariant:
    def test_each_row_is_drawn_around_its_own_mean_and_spread(self):
        source = _source()
        real = source.view("expr", _IDS[:3])

        drawn = _randomized(RandomizationType.INVARIANT).transform(source, _IDS[:3]).matrix

        assert drawn.shape == real.shape
        assert not np.isin(drawn, real).any()
        np.testing.assert_allclose(drawn.mean(axis=1), real.mean(axis=1), atol=0.3)
        np.testing.assert_allclose(drawn.std(axis=1), real.std(axis=1), rtol=0.3)

    def test_an_entity_gets_the_same_draw_in_every_transform(self):
        source = _source()
        featurizer = _randomized(RandomizationType.INVARIANT)

        whole = featurizer.transform(source, _IDS[:4])
        reversed_part = featurizer.transform(source, _IDS[1:4][::-1])

        np.testing.assert_array_equal(whole.matrix[1:], reversed_part.matrix[::-1])
        assert not np.array_equal(
            whole.matrix, _randomized(RandomizationType.INVARIANT, seed=1).transform(source, _IDS[:4]).matrix
        )

    def test_a_ragged_block_keeps_every_shape_and_a_global_block_is_kept(self):
        output = _randomized(RandomizationType.INVARIANT).transform(_source(), _IDS[:3])

        assert [value.shape for value in output.blocks["ragged"].values] == [(2, _N_FEATURES)] * 3
        np.testing.assert_array_equal(output.blocks["global"].values, np.arange(3.0))

    def test_no_entities_give_the_empty_output(self):
        output = _randomized(RandomizationType.INVARIANT).transform(_source(), _IDS[:0])

        assert output.matrix.shape == (0, _N_FEATURES)


def _both_directions(*bonds: tuple[int, int]) -> np.ndarray:
    return np.array([(i, j) for a, b in bonds for i, j in ((a, b), (b, a))]).T


def _bonds(edge_index: np.ndarray) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in edge_index.T.tolist()}


def _degrees(edge_index: np.ndarray, n_nodes: int) -> list[int]:
    return sorted(np.bincount(edge_index.ravel(), minlength=n_nodes).tolist())


_EDGE_INDEX = _both_directions((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))


class _Graphs(Featurizer):
    """Emits the same eight-atom graph for every drug."""

    side: ClassVar[Side] = Side.DRUG
    input_views: ClassVar[tuple[str, ...]] = ("drug_graph",)

    def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None) -> Self:
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        graphs = np.empty(len(ids), dtype=object)
        graphs[:] = [
            Data(
                x=torch.arange(24, dtype=torch.float32).reshape(8, 3),
                edge_index=torch.as_tensor(_EDGE_INDEX),
                edge_attr=torch.ones(_EDGE_INDEX.shape[1], 2),
            )
            for _ in ids
        ]
        return FeaturizerOutput(ids, {"drug_graph": FeatureBlock(graphs, FeatureFormat.GRAPH)})


@pytest.mark.slow
def test_invariant_rewires_every_graph_keeping_its_degrees_and_draws_its_features() -> None:
    randomization = ViewRandomization(frozenset({"drug_graph"}), RandomizationType.INVARIANT, 0)
    featurizer = RandomizedFeaturizer(_Graphs(), randomization, ("drug_graph",))
    ids = np.array(["d1", "d2"])

    graphs = featurizer.transform(MockFeatureSource({}, side=Side.DRUG), ids).blocks["drug_graph"].values

    for graph in graphs:
        assert graph.x.shape == (8, 3)
        assert graph.x.dtype == torch.float32
        assert not torch.equal(graph.x, torch.arange(24, dtype=torch.float32).reshape(8, 3))
        assert graph.edge_attr.shape == (_EDGE_INDEX.shape[1], 2)
        assert _degrees(graph.edge_index.numpy(), 8) == _degrees(_EDGE_INDEX, 8)
    assert _bonds(graphs[0].edge_index.numpy()) != _bonds(graphs[1].edge_index.numpy())


class TestDegreePreservingRewire:
    def test_preserves_degrees_symmetry_and_edge_count(self):
        result = _degree_preserving_rewire(_EDGE_INDEX, np.random.default_rng(42))

        assert result.shape == _EDGE_INDEX.shape
        assert _degrees(result, 8) == _degrees(_EDGE_INDEX, 8)
        assert {tuple(edge) for edge in result.T.tolist()} == {tuple(edge[::-1]) for edge in result.T.tolist()}
        assert _bonds(result) != _bonds(_EDGE_INDEX)
        assert not np.any(result[0] == result[1])

    def test_is_reproducible_for_a_seed(self):
        first = _degree_preserving_rewire(_EDGE_INDEX, np.random.default_rng(3))
        second = _degree_preserving_rewire(_EDGE_INDEX, np.random.default_rng(3))

        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize(
        "edge_index",
        [_both_directions((0, 1)), _both_directions((0, 1), (1, 2), (2, 3))],
        ids=["single-bond", "no-swap-possible"],
    )
    def test_a_graph_that_cannot_swap_is_unchanged(self, edge_index):
        np.testing.assert_array_equal(_degree_preserving_rewire(edge_index, np.random.default_rng(0)), edge_index)


def test_a_constant_row_is_drawn_with_the_minimum_spread() -> None:
    drawn = _resampled_rows(np.full((5, 10), 3.0), np.random.default_rng(0))

    np.testing.assert_allclose(drawn, 3.0, atol=1e-6)

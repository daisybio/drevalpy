"""View randomization: matrices, graphs and whole datasets."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data import synthetic
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset._randomization import (
    _degree_preserving_rewire,
    _randomize_graph,
    _randomize_matrix,
    _randomize_uns_view,
)
from drevalpy.types.enums import RandomizationType


def _degrees(edge_index: np.ndarray, n_nodes: int) -> list[int]:
    return sorted(np.bincount(edge_index.ravel(), minlength=n_nodes).tolist())


class TestRandomizeMatrix:
    def test_permutation_reorders_the_rows(self):
        data = np.arange(40, dtype=np.float32).reshape(10, 4)

        result = _randomize_matrix(data, np.random.default_rng(42), RandomizationType.PERMUTATION)

        assert {tuple(row) for row in result} == {tuple(row) for row in data}
        assert not np.array_equal(result, data)

    def test_invariant_preserves_row_statistics(self):
        data = np.random.default_rng(0).standard_normal((100, 50)).astype(np.float32)

        result = _randomize_matrix(data, np.random.default_rng(1), RandomizationType.INVARIANT)

        assert result.shape == data.shape
        np.testing.assert_allclose(result.mean(axis=1), data.mean(axis=1), atol=0.5)
        assert not np.array_equal(result, data)

    def test_invariant_handles_zero_std(self):
        data = np.full((5, 10), 3.0, dtype=np.float32)

        result = _randomize_matrix(data, np.random.default_rng(0), RandomizationType.INVARIANT)

        np.testing.assert_allclose(result.mean(axis=1), 3.0, atol=0.1)


def _both_directions(*bonds: tuple[int, int]) -> np.ndarray:
    return np.array([(i, j) for a, b in bonds for i, j in ((a, b), (b, a))]).T


def _bonds(edge_index: np.ndarray) -> set[frozenset[int]]:
    return {frozenset(edge) for edge in edge_index.T.tolist()}


class TestDegreePreservingRewire:
    EDGE_INDEX = _both_directions((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))

    def test_preserves_degrees_symmetry_and_edge_count(self):
        result = _degree_preserving_rewire(self.EDGE_INDEX, np.random.default_rng(42))

        assert result.shape == self.EDGE_INDEX.shape
        assert _degrees(result, 8) == _degrees(self.EDGE_INDEX, 8)
        assert {tuple(edge) for edge in result.T.tolist()} == {tuple(edge[::-1]) for edge in result.T.tolist()}
        assert _bonds(result) != _bonds(self.EDGE_INDEX)

    def test_is_reproducible_for_a_seed(self):
        first = _degree_preserving_rewire(self.EDGE_INDEX, np.random.default_rng(3))
        second = _degree_preserving_rewire(self.EDGE_INDEX, np.random.default_rng(3))

        np.testing.assert_array_equal(first, second)

    def test_single_bond_unchanged(self):
        edge_index = _both_directions((0, 1))

        np.testing.assert_array_equal(_degree_preserving_rewire(edge_index, np.random.default_rng(0)), edge_index)

    def test_a_graph_that_cannot_swap_is_unchanged(self):
        edge_index = _both_directions((0, 1), (1, 2), (2, 3))

        np.testing.assert_array_equal(_degree_preserving_rewire(edge_index, np.random.default_rng(0)), edge_index)

    def test_no_self_loops_introduced(self):
        result = _degree_preserving_rewire(self.EDGE_INDEX, np.random.default_rng(7))

        assert not np.any(result[0] == result[1])


class TestRandomizeGraph:
    def test_keys_kept_node_features_randomized_and_degrees_preserved(self):
        edge_index = TestDegreePreservingRewire.EDGE_INDEX
        graph = {
            "x": np.arange(24, dtype=np.float32).reshape(8, 3),
            "edge_index": edge_index,
            "edge_attr": np.ones((edge_index.shape[1], 2), dtype=np.float32),
        }

        result = _randomize_graph(graph, np.random.default_rng(0))

        assert set(result) == set(graph)
        assert result["x"].shape == graph["x"].shape
        assert not np.array_equal(result["x"], graph["x"])
        assert _degrees(result["edge_index"], 8) == _degrees(edge_index, 8)


class TestRandomizeUnsView:
    def test_a_graph_collection_is_randomized_under_invariant(self):
        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        graphs = {"drug1": {"x": np.ones((3, 2)), "edge_index": edge_index}}
        new_uns: dict = {}

        _randomize_uns_view(graphs, "drug_graphs", new_uns, np.random.default_rng(0), RandomizationType.INVARIANT)

        assert set(new_uns["drug_graphs"]) == {"drug1"}
        assert set(new_uns["drug_graphs"]["drug1"]) == {"x", "edge_index"}

    @pytest.mark.parametrize(
        "data", [{"a": 1, "b": 2, "c": 3}, {"d1": {"x": np.zeros((3, 2))}, "d2": {"x": np.ones(5)}}]
    )
    def test_any_other_dict_gets_its_values_reassigned_to_shuffled_keys(self, data):
        new_uns: dict = {}

        _randomize_uns_view(data, "aux", new_uns, np.random.default_rng(0), RandomizationType.INVARIANT)

        assert set(new_uns["aux"]) == set(data)
        assert [id(value) for value in new_uns["aux"].values()] == [id(value) for value in data.values()]

    def test_a_non_dict_is_skipped(self):
        new_uns: dict = {}

        _randomize_uns_view([1, 2], "aux", new_uns, np.random.default_rng(0), RandomizationType.PERMUTATION)

        assert new_uns == {}


class TestWithRandomizedViews:
    @pytest.mark.parametrize("views", [["no_such_view"], ["gene_expression", "no_such_view"]])
    def test_an_unknown_view_raises_key_error(self, synthetic_dataset: Dataset, views):
        with pytest.raises(KeyError, match="no_such_view"):
            synthetic_dataset.with_randomized_views(views)

    def test_an_unknown_randomization_type_is_rejected(self, synthetic_dataset: Dataset):
        with pytest.raises(ValueError, match="not a valid RandomizationType"):
            synthetic_dataset.with_randomized_views(["gene_expression"], randomization_type="shuffle")  # type: ignore[arg-type]

    @pytest.mark.parametrize("randomization_type", list(RandomizationType))
    def test_the_rest_of_the_modality_survives(self, randomization_type):
        dataset = synthetic(n_samples=6, n_drugs=3, omics=["gene_expression"])
        original = dataset.mdata.mod["gene_expression"]
        counts = np.asarray(original.X).copy()
        original.layers["counts"] = counts
        embedding = np.ones((original.n_obs, 2), dtype=np.float32)
        original.obsm["embedding"] = embedding
        original.uns["expression_unit"] = "tpm"

        randomized = dataset.with_randomized_views(
            ["gene_expression"], randomization_type=randomization_type, random_state=0
        ).mdata.mod["gene_expression"]

        assert not np.array_equal(randomized.X, original.X)
        assert randomized.layers is not None
        assert randomized.obsm is not None
        np.testing.assert_array_equal(randomized.layers["counts"], counts)
        np.testing.assert_array_equal(randomized.obsm["embedding"], embedding)
        assert randomized.uns["expression_unit"] == "tpm"
        assert list(randomized.obs_names) == list(original.obs_names)

    def test_the_label_and_name_are_kept_in_the_copy(self, synthetic_dataset: Dataset):
        randomized = synthetic_dataset.with_randomized_views(
            ["gene_expression"], randomization=("SVRC", "gene_expression")
        )

        assert randomized.name == synthetic_dataset.name
        assert randomized.randomization == ("SVRC", "gene_expression")
        assert synthetic_dataset.randomization is None

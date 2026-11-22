"""View randomization: matrices, graphs and whole datasets."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data import synthetic
from drevalpy.data._synthetic_drugs import DRUG_VIEWS, UNS_PAYLOADS
from drevalpy.data._synthetic_omics import SAMPLE_VIEWS
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

    def test_arrays_that_are_not_matrices_are_copied(self):
        graph = {"edge_index": TestDegreePreservingRewire.EDGE_INDEX, "node_ids": np.arange(8), "name": np.array(3.0)}

        result = _randomize_graph(graph, np.random.default_rng(0))

        np.testing.assert_array_equal(result["node_ids"], graph["node_ids"])
        assert result["node_ids"] is not graph["node_ids"]
        assert result["name"] == 3.0


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


def _rows(matrix: np.ndarray) -> list[tuple[float, ...]]:
    return sorted(map(tuple, np.asarray(matrix).tolist()))


@pytest.fixture(scope="module")
def auxiliary_dataset() -> Dataset:
    """A small dataset with drug views in ``varm``, a sample view in ``obsm`` and every ``uns`` payload; not mutated."""
    return synthetic(
        n_samples=6,
        n_drugs=3,
        omics=["gene_expression"],
        sample_views=SAMPLE_VIEWS,
        drug_views=DRUG_VIEWS,
        uns_payloads=UNS_PAYLOADS,
        seed=0,
    )


class TestWithRandomizedAuxiliaryViews:
    """Drug views in ``varm``, sample views in ``obsm`` and payloads in ``uns``, through the public method."""

    @pytest.mark.parametrize("view", ["morgan_fingerprint", "chemberta"])
    def test_permutation_reorders_the_rows_of_a_varm_view_and_leaves_the_others(self, auxiliary_dataset: Dataset, view):
        before = auxiliary_dataset.mdata.mod["response"].varm
        original = {key: np.array(value) for key, value in before.items()}

        after = auxiliary_dataset.with_randomized_views([view], random_state=0).mdata.mod["response"].varm

        assert _rows(after[view]) == _rows(original[view])
        assert not np.array_equal(after[view], original[view])
        for untouched in set(original) - {view}:
            np.testing.assert_array_equal(after[untouched], original[untouched])
        np.testing.assert_array_equal(before[view], original[view])

    def test_invariant_resamples_a_varm_view_around_its_row_statistics(self, auxiliary_dataset: Dataset):
        original = np.array(auxiliary_dataset.mdata.mod["response"].varm["chemberta"])

        randomized = (
            auxiliary_dataset.with_randomized_views(["chemberta"], RandomizationType.INVARIANT, random_state=0)
            .mdata.mod["response"]
            .varm["chemberta"]
        )

        assert randomized.shape == original.shape
        assert not np.array_equal(randomized, original)
        np.testing.assert_allclose(randomized.mean(axis=1), original.mean(axis=1), atol=0.3)
        np.testing.assert_allclose(randomized.std(axis=1), original.std(axis=1), rtol=0.3)

    def test_an_obsm_view_is_permuted(self, auxiliary_dataset: Dataset):
        original = np.array(auxiliary_dataset.mdata.mod["response"].obsm["pathway_features"])

        after = auxiliary_dataset.with_randomized_views(["pathway_features"], random_state=0).mdata.mod["response"]

        assert _rows(after.obsm["pathway_features"]) == _rows(original)
        assert not np.array_equal(after.obsm["pathway_features"], original)
        assert "pathway_features" not in after.varm

    @pytest.mark.parametrize("randomization_type", list(RandomizationType))
    def test_a_non_dict_uns_payload_is_kept_and_warned_about(
        self, auxiliary_dataset: Dataset, randomization_type, caplog
    ):
        with caplog.at_level("WARNING", logger="drevalpy"):
            randomized = auxiliary_dataset.with_randomized_views(["bpe_codes"], randomization_type, random_state=0)

        assert randomized.mdata.uns["bpe_codes"] == auxiliary_dataset.mdata.uns["bpe_codes"]
        assert "Cannot randomize uns key 'bpe_codes' (not a dict)" in caplog.text

    @pytest.mark.parametrize("randomization_type", list(RandomizationType))
    def test_the_values_of_a_dict_payload_move_to_shuffled_keys(self, auxiliary_dataset: Dataset, randomization_type):
        original = auxiliary_dataset.mdata.uns["sparsego"]

        results = [
            auxiliary_dataset.with_randomized_views(["sparsego"], randomization_type, random_state=seed).mdata.uns[
                "sparsego"
            ]
            for seed in range(4)
        ]

        for shuffled in results:
            assert set(shuffled) == set(original)
            assert sorted(shuffled.values()) == sorted(original.values())
        assert any(shuffled != original for shuffled in results)
        assert auxiliary_dataset.mdata.uns["sparsego"] == original

    def test_permutation_reassigns_the_drug_graphs_to_other_drugs(self, auxiliary_dataset: Dataset):
        original = auxiliary_dataset.mdata.uns["drug_graphs"]

        shuffled = auxiliary_dataset.with_randomized_views(["drug_graph"], random_state=0).mdata.uns["drug_graphs"]

        assert set(shuffled) == set(original)
        assert any(not np.array_equal(shuffled[drug]["x"], original[drug]["x"]) for drug in original)
        assert sorted(len(graph["x"]) for graph in shuffled.values()) == sorted(
            len(graph["x"]) for graph in original.values()
        )

    def test_invariant_keeps_each_drug_graph_topology_degrees_and_shapes(self, auxiliary_dataset: Dataset):
        original = auxiliary_dataset.mdata.uns["drug_graphs"]

        randomized = auxiliary_dataset.with_randomized_views(
            ["drug_graph"], RandomizationType.INVARIANT, random_state=0
        ).mdata.uns["drug_graphs"]

        assert list(randomized) == list(original)
        for drug, graph in original.items():
            result = randomized[drug]
            assert {key: value.shape for key, value in result.items()} == {
                key: value.shape for key, value in graph.items()
            }
            n_atoms = graph["x"].shape[0]
            assert _degrees(result["edge_index"], n_atoms) == _degrees(graph["edge_index"], n_atoms)
        assert any(not np.array_equal(randomized[drug]["x"], original[drug]["x"]) for drug in original)

    def test_the_same_seed_gives_the_same_result(self, auxiliary_dataset: Dataset):
        first = auxiliary_dataset.with_randomized_views(["chemberta", "drug_graph"], random_state=5)
        second = auxiliary_dataset.with_randomized_views(["chemberta", "drug_graph"], random_state=5)

        np.testing.assert_array_equal(
            first.mdata.mod["response"].varm["chemberta"], second.mdata.mod["response"].varm["chemberta"]
        )
        assert list(first.mdata.uns["drug_graphs"]) == list(second.mdata.uns["drug_graphs"])

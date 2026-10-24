"""Tests for randomization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.dataset_utils.randomization import (
    _degree_preserving_rewire,
    _is_graph_dict,
    _randomize_graph,
    _randomize_matrix,
)


def _degrees(edge_index: np.ndarray, n_nodes: int) -> list[int]:
    return sorted(np.bincount(edge_index.ravel(), minlength=n_nodes).tolist())


class TestRandomizeMatrix:
    def test_permutation_reorders_the_rows(self):
        data = np.arange(40, dtype=np.float32).reshape(10, 4)

        result = _randomize_matrix(data, np.random.default_rng(42), "permutation")

        assert {tuple(row) for row in result} == {tuple(row) for row in data}
        assert not np.array_equal(result, data)

    def test_invariant_preserves_row_statistics(self):
        data = np.random.default_rng(0).standard_normal((100, 50)).astype(np.float32)

        result = _randomize_matrix(data, np.random.default_rng(1), "invariant")

        assert result.shape == data.shape
        np.testing.assert_allclose(result.mean(axis=1), data.mean(axis=1), atol=0.5)
        assert not np.array_equal(result, data)

    def test_invariant_handles_zero_std(self):
        data = np.full((5, 10), 3.0, dtype=np.float32)

        result = _randomize_matrix(data, np.random.default_rng(0), "invariant")

        np.testing.assert_allclose(result.mean(axis=1), 3.0, atol=0.1)


class TestDegreePreservingRewire:
    def test_preserves_degree_sequence(self):
        edge_index = np.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]])

        result = _degree_preserving_rewire(edge_index, np.random.default_rng(42))

        assert result.shape == edge_index.shape
        assert _degrees(result, 5) == _degrees(edge_index, 5)

    def test_single_edge_unchanged(self):
        edge_index = np.array([[0], [1]])

        np.testing.assert_array_equal(_degree_preserving_rewire(edge_index, np.random.default_rng(0)), edge_index)

    def test_no_self_loops_introduced(self):
        result = _degree_preserving_rewire(np.array([[0, 1, 2, 3], [1, 2, 3, 0]]), np.random.default_rng(7))

        assert not np.any(result[0] == result[1])


class TestRandomizeGraph:
    def test_keys_kept_node_features_randomized_and_degrees_preserved(self):
        edge_index = np.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]])
        graph = {
            "x": np.arange(15, dtype=np.float32).reshape(5, 3),
            "edge_index": edge_index,
            "edge_attr": np.ones((5, 2), dtype=np.float32),
        }

        result = _randomize_graph(graph, np.random.default_rng(0))

        assert set(result) == set(graph)
        assert result["x"].shape == graph["x"].shape
        assert not np.array_equal(result["x"], graph["x"])
        assert _degrees(result["edge_index"], 5) == _degrees(edge_index, 5)


class TestIsGraphDict:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {
                    "drug1": {"x": np.zeros((3, 2)), "edge_index": np.array([[0], [1]])},
                    "drug2": {"x": np.zeros((2, 2)), "edge_index": np.array([[0], [1]])},
                },
                True,
            ),
            ({"a": 1, "b": 2, "c": 3}, False),
            ({"drug1": {"x": np.zeros((3, 2)), "features": np.zeros(5)}}, False),
        ],
        ids=["graph-collection", "plain-dict", "nested-without-edge-index"],
    )
    def test_detection(self, data, expected):
        assert _is_graph_dict(data) is expected


class TestOrchestratorPassesType:
    def test_orchestrator_passes_type(self):
        from unittest.mock import MagicMock

        from drevalpy.experiment import randomization

        mock_model = MagicMock()
        mock_model.config.sample_views.return_value = ["gene_expression"]
        mock_model.config.drug_views.return_value = []
        mock_dataset = MagicMock()
        mock_dataset.with_randomized_views.return_value = MagicMock(randomization=("SVRC", "gene_expression"))

        randomization(mock_model, mock_dataset, ["SVRC"], randomization_type="invariant", random_state=0)

        mock_dataset.with_randomized_views.assert_called_once()
        args, kwargs = mock_dataset.with_randomized_views.call_args
        assert kwargs.get("randomization_type", args[1] if len(args) > 1 else None) == "invariant"


class TestMissingView:
    @pytest.mark.parametrize("views", [["no_such_view"], ["gene_expression", "no_such_view"]])
    def test_an_unknown_view_raises_key_error(self, synthetic_dataset, views):
        with pytest.raises(KeyError, match="no_such_view"):
            synthetic_dataset.with_randomized_views(views)

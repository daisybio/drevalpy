"""Tests for randomization utilities."""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.dataset_utils.randomization import (
    _degree_preserving_rewire,
    _is_graph_dict,
    _randomize_graph,
    _randomize_matrix,
)


class TestRandomizeMatrix:
    """Tests for _randomize_matrix."""

    def test_permutation_preserves_rows(self):
        rng = np.random.default_rng(0)
        data = np.arange(20, dtype=np.float32).reshape(5, 4)
        result = _randomize_matrix(data, rng, "permutation")

        assert result.shape == data.shape
        # Rows are a permutation of the original
        original_set = {tuple(row) for row in data}
        result_set = {tuple(row) for row in result}
        assert original_set == result_set

    def test_permutation_changes_order(self):
        rng = np.random.default_rng(42)
        data = np.arange(40, dtype=np.float32).reshape(10, 4)
        result = _randomize_matrix(data, rng, "permutation")
        assert not np.array_equal(result, data)

    def test_invariant_preserves_row_statistics(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 50)).astype(np.float32)

        result = _randomize_matrix(data, np.random.default_rng(1), "invariant")

        assert result.shape == data.shape
        # Per-row means should match approximately
        np.testing.assert_allclose(result.mean(axis=1), data.mean(axis=1), atol=0.5)
        # Content should be different
        assert not np.array_equal(result, data)

    def test_invariant_handles_zero_std(self):
        rng = np.random.default_rng(0)
        data = np.ones((5, 10), dtype=np.float32) * 3.0
        result = _randomize_matrix(data, rng, "invariant")
        assert result.shape == data.shape
        # With near-zero std, values should be close to the mean
        np.testing.assert_allclose(result.mean(axis=1), 3.0, atol=0.1)


class TestDegreePreservingRewire:
    """Tests for _degree_preserving_rewire."""

    def test_preserves_degree_sequence(self):
        rng = np.random.default_rng(42)
        edge_index = np.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]])
        result = _degree_preserving_rewire(edge_index, rng)

        assert result.shape == edge_index.shape
        # Degree sequence (in+out per node) must be preserved
        max_node = max(edge_index.max(), result.max())
        orig_deg = np.zeros(max_node + 1, dtype=int)
        new_deg = np.zeros(max_node + 1, dtype=int)
        for i in range(edge_index.shape[1]):
            orig_deg[edge_index[0, i]] += 1
            orig_deg[edge_index[1, i]] += 1
        for i in range(result.shape[1]):
            new_deg[result[0, i]] += 1
            new_deg[result[1, i]] += 1
        np.testing.assert_array_equal(sorted(orig_deg), sorted(new_deg))

    def test_single_edge_unchanged(self):
        rng = np.random.default_rng(0)
        edge_index = np.array([[0], [1]])
        result = _degree_preserving_rewire(edge_index, rng)
        np.testing.assert_array_equal(result, edge_index)

    def test_no_self_loops_introduced(self):
        rng = np.random.default_rng(7)
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        result = _degree_preserving_rewire(edge_index, rng)
        assert not np.any(result[0] == result[1])


class TestRandomizeGraph:
    """Tests for _randomize_graph."""

    def test_preserves_structure_keys(self):
        rng = np.random.default_rng(0)
        graph = {
            "x": np.random.randn(5, 3).astype(np.float32),
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]]),
            "edge_attr": np.random.randn(3, 2).astype(np.float32),
        }
        result = _randomize_graph(graph, rng)
        assert set(result.keys()) == set(graph.keys())

    def test_node_features_randomized(self):
        rng = np.random.default_rng(0)
        graph = {
            "x": np.arange(15, dtype=np.float32).reshape(5, 3),
            "edge_index": np.array([[0, 1], [1, 2]]),
        }
        result = _randomize_graph(graph, rng)
        assert not np.array_equal(result["x"], graph["x"])
        assert result["x"].shape == graph["x"].shape

    def test_edge_index_degree_preserved(self):
        rng = np.random.default_rng(0)
        edge_index = np.array([[0, 0, 1, 2, 3], [1, 2, 3, 3, 4]])
        graph = {"x": np.ones((5, 2), dtype=np.float32), "edge_index": edge_index}
        result = _randomize_graph(graph, rng)

        orig_out_degree = np.bincount(edge_index[0])
        new_out_degree = np.bincount(result["edge_index"][0], minlength=len(orig_out_degree))
        np.testing.assert_array_equal(sorted(orig_out_degree), sorted(new_out_degree))


class TestIsGraphDict:
    """Tests for _is_graph_dict."""

    def test_graph_collection_detected(self):
        data = {
            "drug1": {"x": np.zeros((3, 2)), "edge_index": np.array([[0], [1]])},
            "drug2": {"x": np.zeros((2, 2)), "edge_index": np.array([[0], [1]])},
        }
        assert _is_graph_dict(data) is True

    def test_plain_dict_not_detected(self):
        data = {"a": 1, "b": 2, "c": 3}
        assert _is_graph_dict(data) is False

    def test_nested_dict_without_edge_index(self):
        data = {"drug1": {"x": np.zeros((3, 2)), "features": np.zeros(5)}}
        assert _is_graph_dict(data) is False


class TestOrchestratorPassesType:
    """Test that the randomization orchestrator threads randomization_type."""

    def test_orchestrator_passes_type(self, monkeypatch):
        from unittest.mock import MagicMock

        from drevalpy.experiment.randomization import randomization

        mock_config = MagicMock()
        mock_config.cell_line_views.return_value = ["gene_expression"]
        mock_config.drug_views.return_value = []

        mock_model = MagicMock()
        mock_model.model_config.return_value = mock_config

        mock_dataset = MagicMock()
        mock_dataset.with_randomized_views.return_value = MagicMock(randomization=("SVRC", "gene_expression"))

        randomization(mock_model, mock_dataset, ["SVRC"], randomization_type="invariant", random_state=0)

        mock_dataset.with_randomized_views.assert_called_once()
        call_kwargs = mock_dataset.with_randomized_views.call_args
        assert call_kwargs.kwargs.get("randomization_type") or call_kwargs[1].get("randomization_type") is None
        # Check positional or keyword
        args, kwargs = call_kwargs
        if "randomization_type" in kwargs:
            assert kwargs["randomization_type"] == "invariant"
        else:
            assert args[1] == "invariant"

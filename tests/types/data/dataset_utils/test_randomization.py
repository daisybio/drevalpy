"""Randomization, hyperparameter sampling and the stored dataset identity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from drevalpy.types.data.dataset_utils._stored_identity import plain_value, stored_randomization
from drevalpy.types.data.dataset_utils.randomization import (
    _degree_preserving_rewire,
    _is_graph_dict,
    _randomize_graph,
    _randomize_matrix,
)
from drevalpy.types.data.dataset_utils.sampling import _sample_hp_configs


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


def _featurizer(space: dict) -> type:
    """Build a throwaway class exposing *space* as its hyperparameter space."""
    return type("StubFeaturizer", (), {"get_hyperparameter_space": classmethod(lambda cls: space)})


_CATEGORICAL = _featurizer({"scaler": {"type": "categorical", "choices": ["standard", "minmax"]}})
_MIXED = _featurizer(
    {
        "n_components": {"type": "int", "low": 2, "high": 4},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "hidden_dim": {"type": "pow2", "low": 3, "high": 5},
    }
)


class TestEmptySpace:
    def test_a_featurizer_without_hyperparameters_yields_empty_configs(self):
        assert _sample_hp_configs(_featurizer({}), 2) == [{}, {}]

    def test_requesting_zero_configs_from_an_empty_space_yields_nothing(self):
        assert _sample_hp_configs(_featurizer({}), 0) == []


class TestSampling:
    def test_the_requested_number_of_configs_is_returned(self):
        configs = _sample_hp_configs(_CATEGORICAL, 2)

        assert len(configs) == 2

    def test_every_config_covers_the_whole_space(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert set(config) == {"n_components", "learning_rate", "hidden_dim"}

    def test_categorical_values_come_from_the_declared_choices(self):
        configs = _sample_hp_configs(_CATEGORICAL, 2)

        assert all(config["scaler"] in {"standard", "minmax"} for config in configs)

    @pytest.mark.parametrize(
        ("key", "low", "high"),
        [
            pytest.param("n_components", 2, 4, id="int-range"),
            pytest.param("learning_rate", 1e-4, 1e-2, id="log-float-range"),
            pytest.param("hidden_dim", 8, 32, id="pow2-range"),
        ],
    )
    def test_drawn_values_respect_the_declared_bounds(self, key, low, high):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert low <= config[key] <= high

    def test_integer_parameters_are_drawn_as_integers(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert isinstance(config["n_components"], int)

    def test_pow2_parameters_are_exponentiated(self):
        (config,) = _sample_hp_configs(_MIXED, 1)

        assert config["hidden_dim"] in {8, 16, 32}

    def test_an_unknown_parameter_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown hyperparameter type"):
            _sample_hp_configs(_featurizer({"weird": {"type": "quaternion"}}), 1)


class TestStoredRandomization:
    @pytest.mark.parametrize(
        "stored",
        [
            pytest.param(["SVRC", "gene_expression"], id="list"),
            pytest.param(("SVRC", "gene_expression"), id="tuple"),
            pytest.param(np.array(["SVRC", "gene_expression"], dtype=object), id="ndarray"),
        ],
    )
    def test_a_pair_comes_back_as_a_tuple_of_strings(self, stored) -> None:
        assert stored_randomization({"randomization": stored}) == ("SVRC", "gene_expression")

    def test_absent_key_is_no_randomization(self) -> None:
        assert stored_randomization({}) is None

    @pytest.mark.parametrize("stored", [["SVRC"], "SVRC", ["a", "b", "c"], None], ids=["one", "str", "three", "none"])
    def test_anything_but_a_pair_is_no_randomization(self, stored) -> None:
        assert stored_randomization({"randomization": stored}) is None


class TestPlainValue:
    def test_arrays_become_lists_and_numpy_scalars_python_scalars(self) -> None:
        stored = {"views": np.array(["a", "b"], dtype=object), "seed": np.int64(3), "nested": {"x": np.float32(0.5)}}

        plain = plain_value(stored)

        assert plain == {"views": ["a", "b"], "seed": 3, "nested": {"x": 0.5}}
        json.dumps(plain)

    def test_an_empty_array_is_an_empty_list(self) -> None:
        assert plain_value(np.array([])) == []

    def test_plain_python_values_pass_through(self) -> None:
        assert plain_value({"type": "permutation", "views": ["a"], "n": 2}) == {
            "type": "permutation",
            "views": ["a"],
            "n": 2,
        }

"""Tests for the SparseGO block predictor.

Training the GO-structured network needs a real ontology, so what is pinned here
is the surface around it: ontology-metadata parsing, active-view resolution, and
the state/guard paths. Those paths carry the deferred ``torch`` and ``networkx``
imports this module relies on (see ``tests/test_import_cost_policy.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.sparsego.predictor import (
    SparseGOPredictor,
    _parse_ontology_metadata,
    _resolve_active_view,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    seed_literature,
    trainable_parameter_vector,
    two_by_two_batch,
)

_LAYER_CONNECTIONS = [np.array([[0, 1], [1, 2]])]
_GENE2ID = {"TP53": 0, "EGFR": 1}


def _batch(sample_block_names: tuple[str, ...], *, metadata: dict | None = None) -> ModelInputBatch:
    """Build a batch carrying the named sample blocks plus fingerprints.

    :param sample_block_names: Sample block names to populate.
    :param metadata: Optional metadata attached to every sample block.
    :returns: Featurized ``ModelInputBatch``.
    """
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    return two_by_two_batch(
        sample_blocks={name: numeric_feature_block(values, metadata=metadata) for name in sample_block_names},
        drug_blocks={"fingerprints": numeric_feature_block(np.eye(2, dtype=np.float32))},
    )


def test_sparsego_predictor_registry_name() -> None:
    ensure_predictor_registered("sparsego")
    assert get_predictor("sparsego") is SparseGOPredictor


def test_default_hyperparameters_describe_the_go_layer_widths() -> None:
    defaults = SparseGOPredictor.get_default_hyperparameters()

    assert {"num_neurons_per_GO", "num_neurons_drug", "drug_dim", "epochs"} <= set(defaults)


def test_hyperparameter_space_is_empty_so_hpo_skips_this_model() -> None:
    assert SparseGOPredictor.get_hyperparameter_space() == {}


class TestFrozenSparseGONumericalReference:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_v2(self, seed: int) -> None:
        assert_frozen_reference("sparsego", seed)


class TestTinySparseGOFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int) -> None:
        metadata = {
            "layer_connections": [
                np.array([["term-1", "gene-1"], ["term-1", "gene-2"], ["term-2", "gene-3"]]),
                np.array([["root", "term-1"], ["root", "term-2"]]),
            ],
            "gene2id_mapping_ont": {"gene-1": 0, "gene-2": 1, "gene-3": 2},
            "ontology_gene_order": ["gene-1", "gene-2", "gene-3"],
        }
        batch = two_by_two_batch(
            sample_blocks={
                "gene_expression": numeric_feature_block(
                    np.array([[0.1, 0.7, 1.4], [1.3, -0.2, 0.4]], dtype=np.float32),
                    metadata=metadata,
                )
            },
            drug_blocks={
                "fingerprints": numeric_feature_block(
                    np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
                )
            },
        )
        hyperparameters = {
            "num_neurons_per_GO": 2,
            "num_neurons_per_final_GO": 2,
            "num_neurons_drug": [3, 2],
            "num_neurons_final": 3,
            "drug_dim": 4,
            "learning_rate": 0.05,
            "momentum": 0.0,
            "decay_rate": 0.0,
            "p_drop_genes": 0.0,
            "p_drop_terms": 0.0,
            "p_drop_drugs": 0.0,
            "p_drop_final": 0.0,
            "epochs": 1,
            "batch_size": 4,
        }

        seed_literature(seed)
        initial_predictor = SparseGOPredictor(hyperparameters)
        initial_predictor._layer_connections = metadata["layer_connections"]
        initial_predictor._gene2id_mapping_ont = metadata["gene2id_mapping_ont"]
        initial_predictor._build_network()
        assert initial_predictor._model is not None
        initial = trainable_parameter_vector(initial_predictor._model)

        seed_literature(seed)
        predictor = SparseGOPredictor(hyperparameters)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)
        predictions = predictor.predict(batch)
        assert np.isfinite(predictions).all()
        assert np.isfinite(np.mean((predictions - batch.response) ** 2))
        assert_training_predictions("sparsego", seed, predictions)

        restored = SparseGOPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(restored.predict(batch), predictions, rtol=1e-6, atol=1e-7)


class TestParseOntologyMetadata:
    def test_precomputed_structures_are_passed_through(self) -> None:
        connections, gene2id, order = _parse_ontology_metadata(
            {
                "layer_connections": _LAYER_CONNECTIONS,
                "gene2id_mapping_ont": _GENE2ID,
                "ontology_gene_order": ["TP53", "EGFR"],
            }
        )

        assert len(connections) == 1
        assert gene2id == _GENE2ID
        assert order == ["TP53", "EGFR"]

    def test_gene_order_must_match_mapping_indices(self) -> None:
        with pytest.raises(ValueError, match="ontology_gene_order"):
            _parse_ontology_metadata(
                {
                    "layer_connections": _LAYER_CONNECTIONS,
                    "gene2id_mapping_ont": _GENE2ID,
                    "ontology_gene_order": ["EGFR", "TP53"],
                }
            )

    def test_a_missing_gene_order_falls_back_to_the_mapping_keys(self) -> None:
        _, _, order = _parse_ontology_metadata(
            {"layer_connections": _LAYER_CONNECTIONS, "gene2id_mapping_ont": _GENE2ID}
        )

        assert order == list(_GENE2ID)

    def test_neither_structures_nor_file_paths_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="pre-computed ontology structures"):
            _parse_ontology_metadata({})

    def test_partial_structures_without_file_paths_is_an_error(self) -> None:
        """``layer_connections`` alone is not enough to build the network."""
        with pytest.raises(ValueError, match="ontology_file"):
            _parse_ontology_metadata({"layer_connections": _LAYER_CONNECTIONS})


class TestResolveActiveView:
    @pytest.mark.parametrize("view", ["gene_expression", "mutations"])
    def test_exactly_one_supported_block_resolves_to_it(self, view: str) -> None:
        assert _resolve_active_view(_batch((view,))) == view

    def test_both_supported_blocks_is_ambiguous_and_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one sample block"):
            _resolve_active_view(_batch(("gene_expression", "mutations")))

    def test_no_supported_block_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one sample block"):
            _resolve_active_view(_batch(("proteomics",)))


class TestGuards:
    def test_building_the_network_without_ontology_metadata_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ontology metadata"):
            SparseGOPredictor()._build_network()

    def test_fit_requires_metadata_on_the_active_block(self) -> None:
        with pytest.raises(ValueError, match="requires ontology metadata"):
            SparseGOPredictor()._fit(_batch(("gene_expression",)))

    def test_predict_before_fit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be fitted before predict"):
            SparseGOPredictor()._predict(_batch(("gene_expression",)))

    def test_is_fitted_is_false_and_state_is_empty_before_training(self) -> None:
        predictor = SparseGOPredictor()

        assert predictor.is_fitted() is False
        assert predictor.get_state() == {}

    def test_set_state_requires_a_payload_blob(self) -> None:
        with pytest.raises(PredictorStateError, match="payload byte blob"):
            SparseGOPredictor().set_state({})

    def test_set_state_rejects_an_undeserializable_payload(self) -> None:
        with pytest.raises(PredictorStateError, match="could not be deserialized"):
            SparseGOPredictor().set_state({"payload": b"not a torch checkpoint"})

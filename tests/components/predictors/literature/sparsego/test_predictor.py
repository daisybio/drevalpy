"""Tests for the SparseGO block predictor."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.sparsego.predictor import (
    SparseGOPredictor,
    _parse_ontology_metadata,
    _resolve_active_view,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.utils.torch_io import save_trusted_mapping
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
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    return two_by_two_batch(
        sample_blocks={name: numeric_feature_block(values, metadata=metadata) for name in sample_block_names},
        drug_blocks={"fingerprints": numeric_feature_block(np.eye(2, dtype=np.float32))},
    )


def test_registry_name_defaults_and_empty_hyperparameter_space() -> None:
    assert get_predictor("sparsego") is SparseGOPredictor
    defaults = SparseGOPredictor.get_default_hyperparameters()
    assert {"num_neurons_per_GO", "num_neurons_drug", "drug_dim", "epochs"} <= set(defaults)
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


def test_precomputed_ontology_structures_are_passed_through() -> None:
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

    _, _, fallback = _parse_ontology_metadata(
        {"layer_connections": _LAYER_CONNECTIONS, "gene2id_mapping_ont": _GENE2ID}
    )
    assert fallback == list(_GENE2ID)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {
                "layer_connections": _LAYER_CONNECTIONS,
                "gene2id_mapping_ont": _GENE2ID,
                "ontology_gene_order": ["EGFR", "TP53"],
            },
            "ontology_gene_order",
        ),
        ({}, "pre-computed ontology structures"),
        ({"layer_connections": _LAYER_CONNECTIONS}, "ontology_file"),
    ],
)
def test_invalid_ontology_metadata_is_rejected(metadata: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_ontology_metadata(metadata)


@pytest.mark.parametrize(
    ("blocks", "view"),
    [(("gene_expression",), "gene_expression"), (("mutations",), "mutations")],
)
def test_exactly_one_supported_block_resolves_to_the_active_view(blocks: tuple[str, ...], view: str) -> None:
    assert _resolve_active_view(_batch(blocks)) == view


@pytest.mark.parametrize("blocks", [("gene_expression", "mutations"), ("proteomics",)])
def test_zero_or_two_supported_blocks_are_rejected(blocks: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="exactly one sample block"):
        _resolve_active_view(_batch(blocks))


def test_guards_before_fit() -> None:
    with pytest.raises(ValueError, match="ontology metadata"):
        SparseGOPredictor()._build_network()
    with pytest.raises(ValueError, match="requires ontology metadata"):
        SparseGOPredictor()._fit(_batch(("gene_expression",)))
    with pytest.raises(ValueError, match="must be fitted before predict"):
        SparseGOPredictor()._predict(_batch(("gene_expression",)))
    assert SparseGOPredictor().is_fitted() is False
    assert SparseGOPredictor().get_state() == {}


def _payload(**entries: object) -> dict[str, bytes]:
    base = {"predictor_hyperparameters": SparseGOPredictor.get_default_hyperparameters()}
    return {"payload": save_trusted_mapping({**base, **entries})}


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "payload byte blob"),
        ({"payload": b"not a torch checkpoint"}, "could not be deserialized"),
        (_payload(preload={}), "missing model_state"),
        (_payload(model_state=b"x"), "missing preload"),
    ],
)
def test_set_state_rejects_malformed_payloads(state: dict, message: str) -> None:
    with pytest.raises(PredictorStateError, match=message):
        SparseGOPredictor().set_state(state)

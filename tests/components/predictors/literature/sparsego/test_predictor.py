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
from tests.components.predictors.literature._helpers import two_by_two_batch

_LAYER_CONNECTIONS = [np.array([[0, 1], [1, 2]])]
_GENE2ID = {"TP53": 0, "EGFR": 1}


def _batch(cell_line_block_names: tuple[str, ...], *, metadata: dict | None = None) -> ModelInputBatch:
    """Build a batch carrying the named cell-line blocks plus fingerprints.

    :param cell_line_block_names: Cell-line block names to populate.
    :param metadata: Optional metadata attached to every cell-line block.
    :returns: Featurized ``ModelInputBatch``.
    """
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    return two_by_two_batch(
        cell_line_blocks={name: numeric_feature_block(values, metadata=metadata) for name in cell_line_block_names},
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


class TestParseOntologyMetadata:
    def test_precomputed_structures_are_passed_through(self) -> None:
        connections, gene2id, order = _parse_ontology_metadata(
            {
                "layer_connections": _LAYER_CONNECTIONS,
                "gene2id_mapping_ont": _GENE2ID,
                "ontology_gene_order": ["EGFR", "TP53"],
            }
        )

        assert len(connections) == 1
        assert gene2id == _GENE2ID
        assert order == ["EGFR", "TP53"]

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
        with pytest.raises(ValueError, match="exactly one cell-line block"):
            _resolve_active_view(_batch(("gene_expression", "mutations")))

    def test_no_supported_block_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one cell-line block"):
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

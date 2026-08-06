"""Tests for predictor class invariants enforced at registration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.block import BlockPredictor
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.registry.featurizer_registry import cell_line_featurizer_registry, drug_featurizer_registry
from drevalpy.components.registry.predictor_registry import predictor_registry
from drevalpy.types.model_scope import ModelScope


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def test_predictor_must_inherit_exactly_one_leaf_interface() -> None:
    with pytest.raises(ValueError, match="exactly one of"):

        @register_predictor(
            "noLeafPred",
            description="missing leaf",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class NoLeafPred:
            pass


def test_matrix_predictor_requires_numeric_contracts() -> None:
    with pytest.raises(ValueError, match="requires numeric_matrix cell_line contract"):

        @register_predictor(
            "graphMatrixPred",
            description="bad matrix",
            cell_line_contract=FeatureFormat.GRAPH,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class GraphMatrixPred(MatrixPredictor):
            pass


def test_single_drug_feature_predictor_requires_identity_routing() -> None:
    with pytest.raises(ValueError, match="routing_drug_featurizer='identity'"):

        @register_predictor(
            "badSingleDrug",
            description="missing identity routing",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class BadSingleDrug(MatrixPredictor):
            scope = ModelScope.SINGLE_DRUG


def test_feature_free_single_drug_skips_identity_routing() -> None:
    @register_predictor(
        "freeSingleDrug",
        description="feature free single drug",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class FreeSingleDrug(FeatureFreePredictor):
        scope = ModelScope.SINGLE_DRUG

    assert predictor_registry.get("freeSingleDrug") is FreeSingleDrug


def test_registered_predictor_scope_defaults_to_multi_drug() -> None:
    @register_predictor(
        "scopedPred",
        description="scope default",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class ScopedPred(BlockPredictor):
        pass

    assert ScopedPred.scope is ModelScope.MULTI_DRUG


def test_register_existing_rejects_invalid_predictor_class() -> None:
    class InvalidRestored:
        description = "invalid"
        tags = frozenset()
        reference = None
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    with pytest.raises(ValueError, match="exactly one of"):
        predictor_registry.register_existing("invalidRestored", InvalidRestored)


def test_register_existing_restores_valid_predictor() -> None:
    @register_predictor(
        "restorablePred",
        description="restorable",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class RestorablePred(BlockPredictor):
        pass

    predictor_registry.clear()
    assert predictor_registry.list_names() == []
    predictor_registry.register_existing("restorablePred", RestorablePred)
    assert predictor_registry.get("restorablePred") is RestorablePred

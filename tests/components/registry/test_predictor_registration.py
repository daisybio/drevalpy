"""Tests for public predictor registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import get_predictor_metadata, register_predictor
from drevalpy.components.registry.featurizer_registry import cell_line_featurizer_registry, drug_featurizer_registry
from drevalpy.components.registry.predictor_registry import predictor_registry


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    predictor_registry.clear()
    yield
    from drevalpy.components.register_builtins import register_builtin_components

    register_builtin_components()


def test_decorator_returns_original_class() -> None:
    @register_predictor(
        "dummyPred",
        description="pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummyPred(FeatureFreePredictor):
        pass

    assert vars(DummyPred)["registry_name"] == "dummyPred"
    assert vars(DummyPred)["cell_line_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert vars(DummyPred)["drug_contract"] == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_predictor_metadata_catalog_shape() -> None:
    @register_predictor(
        "catalogPred",
        description="catalog shape",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class CatalogPred(FeatureFreePredictor):
        pass

    meta = get_predictor_metadata("catalogPred")
    assert meta["input_interface"] == "feature_free"
    assert meta["description"] == "catalog shape"
    assert meta["tags"] == frozenset()
    for dropped in (
        "cell_line_format",
        "drug_format",
        "supported_modes",
        "supported_scopes",
        "supports_early_stopping",
        "requires_drug_featurizer",
        "required_cell_line_views",
        "required_drug_views",
    ):
        assert dropped not in meta


def test_duplicate_predictor_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set cell_line_contract on the class body"):

        @register_predictor(
            "predConflict",
            description="conflict",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class PredConflict:
            cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_duplicate_predictor_drug_class_and_decorator_contract_fails() -> None:
    with pytest.raises(ValueError, match="do not set drug_contract on the class body"):

        @register_predictor(
            "predDrugConflict",
            description="conflict",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
            drug_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class PredDrugConflict:
            drug_contract = FeatureContract(format=FeatureFormat.GRAPH)


def test_predictor_registration_requires_explicit_contracts() -> None:
    with pytest.raises(TypeError, match="contract"):
        register_predictor("noContractPred", description="missing contracts")  # type: ignore[call-arg]

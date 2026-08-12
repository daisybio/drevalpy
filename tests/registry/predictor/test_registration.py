"""Tests for public predictor registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import metadata as get_predictor_metadata
from drevalpy.registry.predictor import register as register_predictor
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


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


def test_predictor_metadata_facade_returns_catalog_metadata() -> None:
    @register_predictor(
        "catalogPred",
        description="catalog shape",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class CatalogPred(FeatureFreePredictor):
        pass

    meta = get_predictor_metadata("catalogPred")
    assert meta["registry"] == "predictors"
    assert meta["name"] == "catalogPred"
    assert meta["input_interface"] == "feature_free"


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

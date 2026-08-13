"""Tests for public predictor registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import metadata as get_predictor_metadata
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcretePredictor(FeatureFreePredictor):
    """Minimal concrete predictor, so registration is not rejected as abstract."""

    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def test_decorator_returns_original_class() -> None:
    @register_predictor(
        "dummyPred",
        description="pred",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DummyPred(_ConcretePredictor):
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
    class CatalogPred(_ConcretePredictor):
        pass

    meta = get_predictor_metadata("catalogPred")
    assert meta["registry"] == "predictors"
    assert meta["name"] == "catalogPred"
    assert meta["input_interface"] == "feature_free"


def test_class_body_contracts_are_accepted() -> None:
    @register_predictor("bodyContractPred", description="declares on the class body")
    class BodyContractPred(_ConcretePredictor):
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert BodyContractPred.cell_line_contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert BodyContractPred.drug_contract == FeatureContract(format=FeatureFormat.GRAPH)


def test_class_body_contracts_accept_the_format_shorthand() -> None:
    @register_predictor("shorthandPred", description="format shorthand on the class body")
    class ShorthandPred(_ConcretePredictor):
        cell_line_contract = FeatureFormat.NUMERIC_MATRIX
        drug_contract = FeatureFormat.RAGGED_SEQUENCE

    assert ShorthandPred.cell_line_contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert ShorthandPred.drug_contract == FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)


def test_an_invalid_class_body_contract_is_rejected() -> None:
    with pytest.raises(TypeError, match="class-body cell_line_contract is invalid"):

        class BadBodyPred(_ConcretePredictor):
            cell_line_contract = "numeric_matrix_but_a_plain_string"


def test_the_decorator_contract_overrides_the_class_body() -> None:
    @register_predictor(
        "overriddenPred",
        description="decorator wins",
        cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        drug_contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class OverriddenPred(_ConcretePredictor):
        cell_line_contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert OverriddenPred.cell_line_contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_a_predictor_declaring_no_contract_anywhere_is_rejected() -> None:
    with pytest.raises(ValueError, match="no cell_line_contract declared"):

        @register_predictor("noContractPred", description="missing contracts")
        class NoContractPred(_ConcretePredictor):
            pass


def test_a_predictor_missing_only_the_drug_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="no drug_contract declared"):

        @register_predictor(
            "halfContractPred",
            description="only the cell-line side",
            cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class HalfContractPred(_ConcretePredictor):
            pass

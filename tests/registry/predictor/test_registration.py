"""Tests for public predictor registration and lookup helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import metadata as get_predictor_metadata
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from tests.registry._helpers import isolated_component_registries

_NUMERIC = FeatureFormat.NUMERIC_MATRIX


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcretePredictor(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def _contract(fmt: FeatureFormat) -> FeatureContract:
    return FeatureContract(format=fmt)


def test_registration_stamps_the_name_and_contracts_and_exposes_catalog_metadata() -> None:
    @register_predictor("dummyPred", description="pred", sample_contract=_NUMERIC, drug_contract=_NUMERIC)
    class DummyPred(_ConcretePredictor):
        pass

    assert vars(DummyPred)["registry_name"] == "dummyPred"
    assert vars(DummyPred)["sample_contract"] == vars(DummyPred)["drug_contract"] == _contract(_NUMERIC)
    meta = get_predictor_metadata("dummyPred")
    assert (meta["registry"], meta["name"], meta["input_interface"]) == ("predictors", "dummyPred", "feature_free")
    assert predictor_registry._display_name == "predictors"


def test_class_body_contracts_are_accepted_including_the_format_shorthand() -> None:
    @register_predictor("bodyContractPred", description="declares on the class body")
    class BodyContractPred(_ConcretePredictor):
        sample_contract = _contract(_NUMERIC)
        drug_contract = _contract(FeatureFormat.GRAPH)

    @register_predictor("shorthandPred", description="format shorthand on the class body")
    class ShorthandPred(_ConcretePredictor):
        sample_contract = _NUMERIC
        drug_contract = FeatureFormat.RAGGED_SEQUENCE

    assert BodyContractPred.drug_contract == _contract(FeatureFormat.GRAPH)
    assert ShorthandPred.sample_contract == _contract(_NUMERIC)
    assert ShorthandPred.drug_contract == _contract(FeatureFormat.RAGGED_SEQUENCE)


def test_the_decorator_contract_overrides_the_class_body() -> None:
    @register_predictor(
        "overriddenPred", description="decorator wins", sample_contract=_NUMERIC, drug_contract=_NUMERIC
    )
    class OverriddenPred(_ConcretePredictor):
        sample_contract = _contract(FeatureFormat.GRAPH)

    assert OverriddenPred.sample_contract == _contract(_NUMERIC)


def test_an_invalid_class_body_contract_is_rejected() -> None:
    with pytest.raises(TypeError, match="class-body sample_contract is invalid"):

        class BadBodyPred(_ConcretePredictor):
            sample_contract = "numeric_matrix_but_a_plain_string"


@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [({}, "sample_contract"), ({"sample_contract": _NUMERIC}, "drug_contract")],
    ids=["neither", "only-sample"],
)
def test_a_predictor_missing_a_contract_is_rejected(kwargs: dict, missing: str) -> None:
    with pytest.raises(ValueError, match=f"no {missing} declared"):
        register_predictor("halfContractPred", description="missing contracts", **kwargs)(
            type("HalfContractPred", (_ConcretePredictor,), {})
        )

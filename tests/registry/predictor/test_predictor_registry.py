"""Tests for predictor registration, lookup and the class invariants enforced at registration."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.registry._component import predictor_component_metadata
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope
from tests.registry._helpers import empty_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> None:
    empty_component_registries()


class _ConcreteMatrix(MatrixPredictor):
    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.float64)


class _ConcreteBlock(Predictor):
    sample_inputs = Inputs.blocks()
    drug_inputs = Inputs.blocks()

    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


class _ConcreteFeatureFree(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def _register(name: str, cls: type) -> type:
    return register_predictor(name, description=name)(cls)


def test_registration_exposes_catalog_metadata() -> None:
    @register_predictor("dummyPred", description="pred")
    class DummyPred(_ConcreteFeatureFree):
        pass

    assert get_predictor("dummyPred") is DummyPred
    assert dict(predictor_registry.metadata("dummyPred")) == {"description": "pred", "tags": frozenset()}
    meta = predictor_component_metadata("predictors", "dummyPred", DummyPred)
    assert (meta["registry"], meta["name"], meta["input_interface"]) == ("predictors", "dummyPred", "feature_free")
    assert predictor_registry.kind == "predictor"


@pytest.mark.parametrize(
    ("body", "missing"),
    [({}, "sample_inputs"), ({"sample_inputs": Inputs.matrix()}, "drug_inputs")],
    ids=["neither", "only-sample"],
)
def test_a_class_missing_an_inputs_declaration_is_rejected(body: dict, missing: str) -> None:
    with pytest.raises(TypeError, match=f"HalfInputsPred.{missing} must be an Inputs"):
        _register(
            "halfInputsPred",
            type("HalfInputsPred", (_ConcreteBlock,), {"sample_inputs": None, "drug_inputs": None, **body}),
        )

    assert "halfInputsPred" not in predictor_registry


def test_a_class_declaring_something_other_than_inputs_is_rejected() -> None:
    with pytest.raises(TypeError, match="BadPred.drug_inputs must be an Inputs"):
        _register("badPred", type("BadPred", (_ConcreteBlock,), {"drug_inputs": "matrix"}))


@pytest.mark.parametrize("base", [_ConcreteMatrix, _ConcreteFeatureFree], ids=["feature", "free"])
def test_single_drug_predictors_declare_only_their_scope(base: type) -> None:
    cls = _register("singleDrugPred", type("SingleDrugPred", (base,), {"scope": ModelScope.SINGLE_DRUG}))

    assert predictor_registry.get("singleDrugPred") is cls
    assert cls.scope is ModelScope.SINGLE_DRUG


def test_registered_predictor_scope_defaults_to_multi_drug() -> None:
    assert _register("scopedPred", type("ScopedPred", (_ConcreteBlock,), {})).scope is ModelScope.MULTI_DRUG


@pytest.mark.parametrize(
    ("base", "hooks"),
    [(Predictor, "_fit, _predict"), (MatrixPredictor, "_fit_matrix, _predict_matrix")],
    ids=["block", "matrix"],
)
def test_a_predictor_missing_its_hooks_is_rejected_naming_the_registry_and_the_name(base: type, hooks: str) -> None:
    body = {"sample_inputs": Inputs.blocks(), "drug_inputs": Inputs.blocks()}
    with pytest.raises(ValueError, match=rf"predictor 'abstractPred' \(AbstractPred\).*does not implement {hooks}"):
        _register("abstractPred", type("AbstractPred", (base,), body))

    assert "abstractPred" not in predictor_registry

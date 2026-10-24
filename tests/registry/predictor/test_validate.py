"""Tests for predictor class invariants enforced at registration."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config._predictor_traits import needs_identity_drug_routing
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope
from tests.registry._helpers import isolated_component_registries

_NUMERIC = FeatureFormat.NUMERIC_MATRIX


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcreteMatrix(MatrixPredictor):
    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        return None

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(len(x), dtype=np.float64)


class _ConcreteBlock(BlockPredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


class _ConcreteFeatureFree(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


def _register(name: str, cls: type, *, sample: FeatureFormat = _NUMERIC) -> type:
    return register_predictor(name, description=name, sample_contract=sample, drug_contract=_NUMERIC)(cls)


def test_predictor_must_inherit_exactly_one_leaf_interface() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        _register("noLeafPred", type("NoLeafPred", (), {}))


def test_matrix_predictor_requires_numeric_contracts() -> None:
    with pytest.raises(ValueError, match="requires numeric_matrix sample contract"):
        _register("graphMatrixPred", type("GraphMatrixPred", (_ConcreteMatrix,), {}), sample=FeatureFormat.GRAPH)


@pytest.mark.parametrize(
    ("base", "routes_identity"), [(_ConcreteMatrix, True), (_ConcreteFeatureFree, False)], ids=["feature", "free"]
)
def test_single_drug_predictors_declare_only_their_scope(base: type, routes_identity: bool) -> None:
    cls = _register("singleDrugPred", type("SingleDrugPred", (base,), {"scope": ModelScope.SINGLE_DRUG}))

    assert predictor_registry.get("singleDrugPred") is cls
    assert needs_identity_drug_routing("singleDrugPred") is routes_identity


def test_registered_predictor_scope_defaults_to_multi_drug() -> None:
    assert _register("scopedPred", type("ScopedPred", (_ConcreteBlock,), {})).scope is ModelScope.MULTI_DRUG


def _restored(base: type = object, **attrs) -> type:
    attributes = {
        "description": "restored",
        "tags": frozenset(),
        "reference": None,
        "sample_contract": FeatureContract(format=_NUMERIC),
        "drug_contract": FeatureContract(format=_NUMERIC),
    }
    return type("Restored", (base,), attributes | attrs)


def test_register_existing_restores_valid_and_rejects_invalid_classes() -> None:
    restorable = _register("restorablePred", type("RestorablePred", (_ConcreteBlock,), {}))
    predictor_registry.clear()
    assert predictor_registry.list_names() == []

    predictor_registry.register_existing("restorablePred", restorable)

    assert predictor_registry.get("restorablePred") is restorable
    with pytest.raises(ValueError, match="exactly one of"):
        predictor_registry.register_existing("invalidRestored", _restored())
    with pytest.raises(ValueError, match="does not implement"):
        predictor_registry.register_existing("abstractRestored", _restored(BlockPredictor))


@pytest.mark.parametrize(
    ("base", "hooks"),
    [(BlockPredictor, "_fit, _predict"), (MatrixPredictor, "_fit_matrix, _predict_matrix")],
    ids=["block", "matrix"],
)
def test_a_predictor_missing_its_hooks_is_rejected_naming_the_registry_and_the_name(base: type, hooks: str) -> None:
    with pytest.raises(ValueError, match=rf"predictor 'abstractPred' \(AbstractPred\).*does not implement {hooks}"):
        _register("abstractPred", type("AbstractPred", (base,), {}))

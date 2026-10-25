"""Tests for predictor registration, lookup and the class invariants enforced at registration."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.models.config._predictor_traits import needs_identity_drug_routing
from drevalpy.registry.predictor import PredictorRegistry, predictor_registry
from drevalpy.registry.predictor import metadata as get_predictor_metadata
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.model_scope import ModelScope
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


def test_registration_stamps_the_name_and_formats_and_exposes_catalog_metadata() -> None:
    @register_predictor("dummyPred", description="pred", sample_input_format=_NUMERIC, drug_input_format=_NUMERIC)
    class DummyPred(_ConcretePredictor):
        pass

    assert vars(DummyPred)["registry_name"] == "dummyPred"
    assert vars(DummyPred)["sample_input_format"] == vars(DummyPred)["drug_input_format"] == _NUMERIC
    meta = get_predictor_metadata("dummyPred")
    assert (meta["registry"], meta["name"], meta["input_interface"]) == ("predictors", "dummyPred", "feature_free")
    assert predictor_registry._display_name == "predictors"


def test_class_body_formats_are_accepted() -> None:
    @register_predictor("bodyFormatPred", description="declares on the class body")
    class BodyFormatPred(_ConcretePredictor):
        sample_input_format = _NUMERIC
        drug_input_format = FeatureFormat.GRAPH

    assert BodyFormatPred.sample_input_format == _NUMERIC
    assert BodyFormatPred.drug_input_format == FeatureFormat.GRAPH


def test_the_decorator_format_overrides_the_class_body() -> None:
    @register_predictor(
        "overriddenPred", description="decorator wins", sample_input_format=_NUMERIC, drug_input_format=_NUMERIC
    )
    class OverriddenPred(_ConcretePredictor):
        sample_input_format = FeatureFormat.GRAPH

    assert OverriddenPred.sample_input_format == _NUMERIC


def test_an_invalid_class_body_format_is_rejected() -> None:
    with pytest.raises(TypeError, match="BadBodyPred.sample_input_format must be a FeatureFormat"):

        class BadBodyPred(_ConcretePredictor):
            sample_input_format = "numeric_matrix_but_a_plain_string"


@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [({}, "sample_input_format"), ({"sample_input_format": _NUMERIC}, "drug_input_format")],
    ids=["neither", "only-sample"],
)
def test_a_predictor_missing_a_format_is_rejected(kwargs: dict, missing: str) -> None:
    with pytest.raises(TypeError, match=f"HalfFormatPred.{missing} must be a FeatureFormat"):
        register_predictor("halfFormatPred", description="missing formats", **kwargs)(
            type("HalfFormatPred", (_ConcretePredictor,), {})
        )


def test_an_isolated_registry_registers_with_formats() -> None:
    registry = PredictorRegistry()
    assert registry._registry_id == "predictor"

    @registry.register(
        "localPred",
        description="local",
        sample_input_format=FeatureFormat.NUMERIC_MATRIX,
        drug_input_format=FeatureFormat.GRAPH,
    )
    class LocalPred(FeatureFreePredictor):
        def _fit(self, batch: ModelInputBatch) -> None:
            return None

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.zeros(batch.n_pairs, dtype=np.float64)

    assert registry.get("localPred") is LocalPred
    assert vars(LocalPred)["sample_input_format"] == FeatureFormat.NUMERIC_MATRIX
    assert vars(LocalPred)["drug_input_format"] == FeatureFormat.GRAPH


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
    return register_predictor(name, description=name, sample_input_format=sample, drug_input_format=_NUMERIC)(cls)


def test_predictor_must_inherit_exactly_one_leaf_interface() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        _register("noLeafPred", type("NoLeafPred", (), {}))


def test_matrix_predictor_requires_numeric_formats() -> None:
    with pytest.raises(ValueError, match="requires numeric_matrix sample input format"):
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
        "sample_input_format": _NUMERIC,
        "drug_input_format": _NUMERIC,
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

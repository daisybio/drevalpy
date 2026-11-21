"""Tests for the shared Predictor contract and the feature-free and matrix bases."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.data._synthetic import synthetic as build_synthetic_dataset
from drevalpy.models.config import PredictorConfig
from drevalpy.registry._component import predictor_component_metadata
from drevalpy.registry.predictor import predictor_registry
from drevalpy.testing import build_synthetic_batch
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors.literature._helpers import two_by_two_batch


class _StubPredictor(Predictor):
    sample_inputs = Inputs.matrix()
    drug_inputs = Inputs.matrix()

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def _fit(self, batch: ModelInputBatch) -> None:
        _ = batch

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


EXPECTED = {
    "feature_free": {"naiveMean"},
    "matrix": {
        "elasticNet",
        "singleDrugElasticNet",
        "lasso",
        "ridge",
        "randomForest",
        "singleDrugRandomForest",
        "singleDrugKnn",
        "singleDrugMorgoth",
        "svr",
        "gradientBoosting",
        "adaboost",
        "knn",
        "xgboost",
        "lightgbm",
        "neuralNetwork",
    },
    "block": {
        "naiveDrugMean",
        "naiveSampleMean",
        "naiveTissueMean",
        "naiveTissueDrugMean",
        "naiveMeanEffects",
        "precily",
        "srmf",
        "drugGNN",
        "dipk",
        "pharmaFormer",
        "sparsego",
        "molir",
        "superfeltr",
        "ensembleMF",
        "paccMann",
    },
}


def _ids(n: int) -> np.ndarray:
    return np.array([f"id_{i}" for i in range(n)])


def test_predictor_init_merges_default_hyperparameters_and_has_no_public_build() -> None:
    predictor = _StubPredictor(hyperparameters={"alpha": 0.5, "extra": True})
    assert predictor._hyperparameters["alpha"] == 0.5
    assert predictor._hyperparameters["extra"] is True
    assert "build" not in Predictor.__dict__
    assert not hasattr(predictor, "build")


def test_predictor_config_create_instance_passes_hyperparameters() -> None:
    predictor = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.25})
    assert isinstance(predictor, ElasticNetPredictor)
    assert predictor._hyperparameters["alpha"] == 0.25
    assert predictor._hyperparameters["alpha"] == 0.25


def test_builtin_predictor_interfaces_partition() -> None:
    observed: dict[str, set[str]] = {"feature_free": set(), "matrix": set(), "block": set()}
    for name, cls in predictor_registry.items():
        observed[predictor_component_metadata("predictors", name, cls)["input_interface"]].add(name)
    assert observed == EXPECTED


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("invalid", ["missing", "response", "features"])
def test_stopping_requirement_is_checked_after_pair_filtering(monkeypatch, required, invalid) -> None:
    values = np.array([[1.0], [np.nan if invalid == "features" else 2.0]])
    batch = two_by_two_batch(sample_blocks={"gene_expression": numeric_feature_block(values)}, drug_blocks={})
    if invalid != "missing":
        batch.early_stopping_response = ResponseBatch(
            np.array([np.nan if invalid == "response" else 3.0]), np.array(["cl2"]), np.array(["d1"])
        )
    monkeypatch.setattr(_StubPredictor, "requires_early_stopping", required)
    if required:
        with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
            _StubPredictor().fit(batch)
    else:
        _StubPredictor().fit(batch)


@pytest.mark.parametrize(("base", "kind"), [(FeatureFreePredictor, "none"), (MatrixPredictor, "matrix")])
def test_bases_declare_the_same_inputs_on_both_sides(base: type[Predictor], kind: str) -> None:
    assert issubclass(base, Predictor)
    assert base.sample_inputs.kind == base.drug_inputs.kind == kind


def test_matrix_base_is_abstract_over_its_matrix_hooks() -> None:
    assert {"_fit_matrix", "_predict_matrix"} <= MatrixPredictor.__abstractmethods__
    assert {"_fit", "_predict"} <= Predictor.__abstractmethods__


def test_a_block_predictor_is_a_predictor_supplying_both_hooks() -> None:
    class _Constant(Predictor):
        sample_inputs = Inputs.blocks()
        drug_inputs = Inputs.none()

        def _fit(self, batch: ModelInputBatch) -> None:
            self._value = 1.0

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.full(len(batch.sample_ids), self._value)

    assert not inspect.isabstract(_Constant)


def test_is_fitted_turns_true_once_fit_completes() -> None:
    predictor = _StubPredictor()
    assert predictor.is_fitted() is False
    predictor.fit(build_synthetic_batch(build_synthetic_dataset()))
    assert predictor.is_fitted() is True


def test_matrix_predictor_cannot_build_a_design_matrix_without_a_response() -> None:
    class _Recording(MatrixPredictor):
        def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
            raise AssertionError("should not be reached")

        def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    batch = ModelInputBatch(
        sample_ids=np.array(["e0", "e1"]),
        drug_ids=_ids(2),
        response=None,
        sample=FeaturizerResult.from_blocks({"view": numeric_feature_block(np.ones((2, 1)))}, np.array(["e0", "e1"])),
        drug=None,
        sample_pair_idx=np.array([0, 1], dtype=np.int64),
        drug_pair_idx=None,
    )

    with pytest.raises(RuntimeError, match="batch.response is required for fit"):
        _Recording()._fit(batch)


@pytest.mark.parametrize(
    ("response_length", "pair_idx_length", "message"),
    [(2, 3, "response must be one-dimensional"), (3, 2, "sample_pair_idx must be one-dimensional")],
)
def test_batch_rejects_length_mismatches(response_length: int, pair_idx_length: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelInputBatch(
            sample_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(response_length),
            sample=FeaturizerResult(np.empty(0), {}, ()),
            drug=None,
            sample_pair_idx=np.zeros(pair_idx_length, dtype=np.int64),
            drug_pair_idx=None,
        )

"""Tests for the shared Predictor contract and the feature-free and matrix bases."""

from __future__ import annotations

import inspect
from dataclasses import replace

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
from drevalpy.testing import build_synthetic_input
from drevalpy.types.data.feature_block import FeatureBlock
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from tests.components.predictors.literature._helpers import two_by_two_batch


class _StubPredictor(Predictor):
    sample_inputs = Inputs.matrix()
    drug_inputs = Inputs.matrix()

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def _fit(self, batch: PredictorInput) -> None:
        _ = batch

    def _predict(self, batch: PredictorInput) -> np.ndarray:
        return np.zeros(len(batch), dtype=np.float64)


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
    batch = two_by_two_batch(sample_blocks={"gene_expression": FeatureBlock(values)}, drug_blocks={})
    if invalid != "missing":
        batch = replace(
            batch,
            early_stopping=ResponseTable(
                np.array([np.nan if invalid == "response" else 3.0]), np.array(["cl2"]), np.array(["d1"])
            ),
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

        def _fit(self, batch: PredictorInput) -> None:
            self._value = 1.0

        def _predict(self, batch: PredictorInput) -> np.ndarray:
            return np.full(len(batch.sample_ids), self._value)

    assert not inspect.isabstract(_Constant)


def test_is_fitted_turns_true_once_fit_completes() -> None:
    predictor = _StubPredictor()
    assert predictor.is_fitted() is False
    predictor.fit(build_synthetic_input(build_synthetic_dataset()))
    assert predictor.is_fitted() is True

"""Tests for the shared Predictor constructor contract and the input-interface taxonomy."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.models.config import PredictorConfig
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor import list as list_predictors
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class _StubPredictor(Predictor):
    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def _fit(self, batch: ModelInputBatch) -> None:
        _ = batch

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


_StubPredictor.cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
_StubPredictor.drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

EXPECTED = {
    "feature_free": {"naiveMean"},
    "matrix": {
        "elasticNet",
        "singleDrugElasticNet",
        "lasso",
        "ridge",
        "randomForest",
        "singleDrugRandomForest",
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
        "naiveCellLineMean",
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
    },
}
LEAF_BASES = (FeatureFreePredictor, MatrixPredictor, BlockPredictor)


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_components()


def test_predictor_accepts_class_body_contracts() -> None:
    class BodyContractPredictor(Predictor):  # noqa: B903
        cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    assert BodyContractPredictor.cell_line_contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def test_predictor_normalizes_a_class_body_format_shorthand() -> None:
    class ShorthandPredictor(Predictor):  # noqa: B903
        drug_contract = FeatureFormat.GRAPH

    assert ShorthandPredictor.drug_contract == FeatureContract(format=FeatureFormat.GRAPH)


def test_predictor_rejects_an_invalid_class_body_contract() -> None:
    with pytest.raises(TypeError, match="class-body drug_contract is invalid"):

        class BadPredictor(Predictor):  # noqa: B903
            drug_contract = "graph_but_a_plain_string"


def test_predictor_init_merges_default_hyperparameters() -> None:
    predictor = _StubPredictor(hyperparameters={"alpha": 0.5, "extra": True})
    assert predictor._hyperparameters["alpha"] == 0.5
    assert predictor._hyperparameters["extra"] is True


def test_predictor_has_no_public_build() -> None:
    assert "build" not in Predictor.__dict__
    assert not hasattr(_StubPredictor(), "build")


def test_predictor_config_create_instance_passes_hyperparameters() -> None:
    register_builtin_components()
    predictor = PredictorConfig(name="elasticNet").create_instance({"alpha": 0.25})
    assert isinstance(predictor, ElasticNetPredictor)
    assert predictor._hyperparameters["alpha"] == 0.25
    assert predictor._h["alpha"] == 0.25


def test_interface_bases_declare_input_interface() -> None:
    assert FeatureFreePredictor.input_interface == "feature_free"
    assert MatrixPredictor.input_interface == "matrix"
    assert BlockPredictor.input_interface == "block"


def test_builtin_predictor_interfaces_partition() -> None:
    observed: dict[str, set[str]] = {
        "feature_free": set(),
        "matrix": set(),
        "block": set(),
    }
    for name in list_predictors():
        cls = get_predictor(name)
        matches = [base for base in LEAF_BASES if issubclass(cls, base)]
        assert len(matches) == 1, (name, matches)
        assert cls.input_interface == matches[0].input_interface
        observed[cls.input_interface].add(name)
    assert observed == EXPECTED

"""Tests for the shared Predictor contract and the feature-free, matrix and block input interfaces."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor
from drevalpy.models.config import PredictorConfig
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor import list as list_predictors
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors.literature._helpers import two_by_two_batch


class _StubPredictor(Predictor):
    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, object]]:
        return {"alpha": {"type": "float", "default": 1.0}}

    def _fit(self, batch: ModelInputBatch) -> None:
        _ = batch

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)


_StubPredictor.sample_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
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
        "singleDrugKnn",
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
LEAF_BASES = (FeatureFreePredictor, MatrixPredictor, BlockPredictor)


def _ids(n: int) -> np.ndarray:
    return np.array([f"id_{i}" for i in range(n)])


def test_predictor_normalizes_class_body_contracts_and_rejects_invalid_ones() -> None:
    class BodyContractPredictor(Predictor):
        sample_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
        drug_contract = FeatureFormat.GRAPH

    assert BodyContractPredictor.sample_contract == FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert BodyContractPredictor.drug_contract == FeatureContract(format=FeatureFormat.GRAPH)
    with pytest.raises(TypeError, match="class-body drug_contract is invalid"):

        class BadPredictor(Predictor):
            drug_contract = "graph_but_a_plain_string"


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
    assert predictor._h["alpha"] == 0.25


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


@pytest.mark.parametrize(
    ("base", "interface"),
    [(FeatureFreePredictor, "feature_free"), (MatrixPredictor, "matrix"), (BlockPredictor, "block")],
)
def test_leaf_bases_declare_their_interface_under_predictor(base: type[Predictor], interface: str) -> None:
    assert issubclass(base, Predictor)
    assert base.input_interface == interface
    assert not issubclass(base, tuple(other for other in LEAF_BASES if other is not base))


def test_matrix_and_block_bases_are_abstract_over_their_own_hooks() -> None:
    assert {"_fit_matrix", "_predict_matrix"} <= MatrixPredictor.__abstractmethods__
    assert {"_fit", "_predict"} <= BlockPredictor.__abstractmethods__
    with pytest.raises(TypeError):
        BlockPredictor()


def test_block_predictor_subclass_supplying_both_hooks_is_concrete() -> None:
    class _Constant(BlockPredictor):
        def _fit(self, batch: ModelInputBatch) -> None:
            self._value = 1.0

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.full(len(batch.sample_ids), self._value)

    assert not inspect.isabstract(_Constant)
    assert _Constant.input_interface == "block"


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

"""Tests for the shared Predictor contract and the feature-free and matrix bases."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
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


class _RecordingPredictor(_StubPredictor):
    """Scores every pair 7.0 and records what ``_fit`` and ``_predict`` were given."""

    def __init__(self) -> None:
        super().__init__()
        self.fitted_on: list[PredictorInput] = []
        self.predicted_on: list[PredictorInput] = []

    def _fit(self, batch: PredictorInput) -> None:
        self.fitted_on.append(batch)

    def _predict(self, batch: PredictorInput) -> np.ndarray:
        self.predicted_on.append(batch)
        return np.full(len(batch), 7.0)


def _batch_with_sample_values(
    values: np.ndarray, block_format: FeatureFormat = FeatureFormat.NUMERIC_MATRIX
) -> PredictorInput:
    """Four pairs, ``cl1`` x (``d1``, ``d2``) then ``cl2`` x (``d1``, ``d2``), with one block row per sample."""
    return two_by_two_batch(sample_blocks={"gene_expression": FeatureBlock(values, block_format)}, drug_blocks={})


class TestPairsWithMissingFeatures:
    def test_predict_returns_nan_for_pairs_with_nan_features_and_scores_the_rest(self) -> None:
        batch = _batch_with_sample_values(np.array([[1.0], [np.nan]]))
        predictor = _RecordingPredictor()

        predictions = predictor.predict(batch)

        np.testing.assert_array_equal(predictions, [7.0, 7.0, np.nan, np.nan])
        assert [len(scored) for scored in predictor.predicted_on] == [2]
        assert list(predictor.predicted_on[0].sample_ids) == ["cl1", "cl1"]

    def test_predict_does_not_call_the_model_when_every_pair_is_invalid(self) -> None:
        batch = _batch_with_sample_values(np.full((2, 1), np.nan))
        predictor = _RecordingPredictor()

        predictions = predictor.predict(batch)

        assert predictions.shape == (4,)
        assert np.isnan(predictions).all()
        assert predictor.predicted_on == []

    def test_predict_passes_a_fully_valid_input_on_unchanged(self) -> None:
        batch = _batch_with_sample_values(np.array([[1.0], [2.0]]))
        predictor = _RecordingPredictor()

        np.testing.assert_array_equal(predictor.predict(batch), [7.0] * 4)
        assert predictor.predicted_on == [batch]

    def test_a_ragged_block_row_holding_none_marks_its_pairs_invalid(self) -> None:
        values = np.empty(2, dtype=object)
        values[0], values[1] = np.ones((3, 2)), None
        batch = _batch_with_sample_values(values, FeatureFormat.RAGGED_SEQUENCE)

        predictions = _RecordingPredictor().predict(batch)

        np.testing.assert_array_equal(predictions, [7.0, 7.0, np.nan, np.nan])

    @pytest.mark.parametrize(
        ("rows", "expected"),
        [
            (np.array([[1, 2], [3, 4]]), [True, True]),
            (np.array([[1.0, np.nan], [3.0, 4.0]]), [False, True]),
            (np.array([[1 + 1j], [complex("nan")]]), [True, False]),
            (np.array([["a", "b"], ["c", "d"]]), [True, True]),
        ],
        ids=["integers", "float-nan", "complex-nan", "text-has-no-missing-marker"],
    )
    def test_valid_feature_rows_by_dtype(self, rows: np.ndarray, expected: list[bool]) -> None:
        np.testing.assert_array_equal(Predictor._valid_feature_rows(rows), expected)

    def test_a_block_that_is_not_aligned_with_the_entities_is_not_checked(self) -> None:
        shared = FeatureBlock(np.array([[np.nan]]))
        batch = two_by_two_batch(
            sample_blocks={"gene_expression": FeatureBlock(np.array([[1.0], [2.0]]))}, drug_blocks={"shared": shared}
        )

        np.testing.assert_array_equal(_RecordingPredictor().predict(batch), [7.0] * 4)

    def test_an_input_without_a_drug_table_is_checked_on_the_sample_side_only(self) -> None:
        batch = replace(_batch_with_sample_values(np.array([[1.0], [np.nan]])), drug=None)

        np.testing.assert_array_equal(_RecordingPredictor().predict(batch), [7.0, 7.0, np.nan, np.nan])

    def test_fit_refuses_an_input_without_any_response(self) -> None:
        batch = _batch_with_sample_values(np.array([[1.0], [2.0]]))
        batch = replace(batch, pairs=replace(batch.pairs, response=np.full(4, np.nan)))
        predictor = _RecordingPredictor()

        with pytest.raises(ValueError, match="Predictors require response values during fit"):
            predictor.fit(batch)
        assert predictor.fitted_on == [] and not predictor.is_fitted()

    def test_an_empty_input_is_fitted_and_predicted_without_a_nan_warning(self, caplog) -> None:
        empty = _batch_with_sample_values(np.array([[1.0], [2.0]]))[np.zeros(4, dtype=bool)]
        predictor = _RecordingPredictor()

        with caplog.at_level("WARNING"):
            predictor.fit(empty)
            predictions = predictor.predict(empty)

        assert len(predictor.fitted_on) == 1 and len(predictor.fitted_on[0]) == 0
        assert predictor.is_fitted()
        assert predictions.shape == (0,)
        assert "NaN features" not in caplog.text

    def test_fit_warns_when_too_many_pairs_are_dropped(self, caplog) -> None:
        batch = _batch_with_sample_values(np.array([[1.0], [np.nan]]))
        predictor = _RecordingPredictor()

        with caplog.at_level("WARNING", logger="drevalpy"):
            predictor.fit(batch)

        assert "_RecordingPredictor.fit: 50% of pairs have NaN features (threshold: 20%)" in caplog.text
        assert len(predictor.fitted_on[0]) == 2


def test_matrix_predictor_hands_the_design_matrix_and_response_to_its_hooks() -> None:
    seen: dict[str, np.ndarray] = {}

    class _Matrix(MatrixPredictor):
        def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
            seen["fit_x"], seen["fit_y"] = x, y

        def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
            seen["predict_x"] = x
            return x.sum(axis=1)

    batch = _batch_with_sample_values(np.array([[1.0, 2.0], [10.0, 20.0]]))
    predictor = _Matrix()

    predictor.fit(batch)
    predictions = predictor.predict(batch)

    expected = np.array([[1.0, 2.0], [1.0, 2.0], [10.0, 20.0], [10.0, 20.0]], dtype=np.float32)
    np.testing.assert_array_equal(seen["fit_x"], expected)
    np.testing.assert_array_equal(seen["fit_y"], batch.response)
    np.testing.assert_array_equal(seen["predict_x"], expected)
    np.testing.assert_array_equal(predictions, [3.0, 3.0, 30.0, 30.0])

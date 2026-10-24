"""Tests for the scikit-learn, LightGBM and XGBoost tabular predictors and their shared bases."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._boosted_trees import SHARED_DEFAULTS, SHARED_SPACE, BoostedTreesPredictor
from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.lightgbm_pred import LightGBMPredictor
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_models import (
    AdaBoostPredictor,
    ElasticNetPredictor,
    GradientBoostingPredictor,
    KNNPredictor,
    LassoPredictor,
    RandomForestPredictor,
    RidgePredictor,
    SingleDrugElasticNetPredictor,
    SingleDrugKNNPredictor,
    SingleDrugRandomForestPredictor,
    SVRPredictor,
)
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.predictors.xgboost_pred import XGBoostPredictor, _set_xgboost_thread_defaults
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models._pipeline import _ModelPipeline
from drevalpy.models.config import ModelConfig, from_spec
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.components.predictors._helpers import neural_batch
from tests.conftest import MockFeatureSource

TINY_LIGHTGBM = {"n_estimators": 3, "max_depth": 2, "num_leaves": 3, "n_jobs": 1}


class _Bare(BoostedTreesPredictor):
    def _make_estimator(self) -> Any:
        return object()


class _Tweaked(BoostedTreesPredictor):
    boosting_default_overrides: ClassVar[dict[str, Any]] = {"subsample": 0.8}
    boosting_extra_defaults: ClassVar[dict[str, Any]] = {"num_leaves": 63}
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {"max_depth": {"high": 8}}
    tuned_hyperparameters: ClassVar[tuple[str, ...]] = ("max_depth", "num_leaves")

    def _make_estimator(self) -> Any:
        return object()


def test_defaults_and_hyperparameter_merging() -> None:
    assert RidgePredictor.supported_modes == frozenset({PredictionMode.REGRESSION})
    assert RidgePredictor()._mode is PredictionMode.REGRESSION
    assert RidgePredictor().is_fitted() is False
    predictor = ElasticNetPredictor(hyperparameters={"max_iter": 7, "alpha": 0.5})

    assert predictor._h["max_iter"] == 7
    assert predictor._h["tol"] == pytest.approx(1e-4)
    assert predictor._h["alpha"] == pytest.approx(0.5)


def test_an_empty_design_matrix_leaves_the_estimator_unset_and_predicts_nan() -> None:
    predictor = RidgePredictor()

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))
    preds = predictor._predict_matrix(np.zeros((3, 2)))

    assert predictor._estimator is None
    assert predictor.is_fitted() is False
    assert preds.shape == (3,)
    assert np.isnan(preds).all()
    assert preds.dtype == np.float64


def test_state_round_trip_reproduces_predictions() -> None:
    predictor = RidgePredictor(hyperparameters={"alpha": 0.25})
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())
    state = predictor.get_state()
    assert set(state) == {"estimator", "hyperparameters", "mode"}
    assert state["mode"] == PredictionMode.REGRESSION.value

    restored = RidgePredictor()
    restored.set_state(state)

    assert restored.is_fitted()
    assert restored._h["alpha"] == pytest.approx(0.25)
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)
    state["mode"] = PredictionMode.REGRESSION
    instance_restored = RidgePredictor()
    instance_restored.set_state(state)
    assert instance_restored._mode is PredictionMode.REGRESSION


@pytest.mark.parametrize(
    ("entry", "value", "message"),
    [
        ("estimator", None, "missing a fitted estimator"),
        ("hyperparameters", {}, "missing hyperparameters"),
        ("mode", 3, "invalid prediction mode"),
    ],
)
def test_set_state_rejects_malformed_state_and_leaves_the_predictor_untouched(
    entry: str, value: object, message: str
) -> None:
    predictor = RidgePredictor(hyperparameters={"alpha": 0.25})
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())
    state = predictor.get_state()
    if value is None:
        del state[entry]
    else:
        state[entry] = value

    with pytest.raises(PredictorStateError, match=message):
        RidgePredictor().set_state(state)
    with pytest.raises(PredictorStateError, match=message):
        predictor.set_state(state)

    assert predictor._h["alpha"] == pytest.approx(0.25)
    np.testing.assert_allclose(predictor.predict(neural_batch()), expected)


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
        (SingleDrugKNNPredictor, KNNPredictor),
    ],
)
def test_single_drug_sklearn_predictors_route_by_identity(
    predictor_class: type[Predictor], shared_predictor_class: type[Predictor]
) -> None:
    assert shared_predictor_class.scope is ModelScope.MULTI_DRUG
    assert issubclass(predictor_class, shared_predictor_class)
    assert issubclass(predictor_class, SingleDrugSklearnPredictor)
    assert predictor_class.scope is ModelScope.SINGLE_DRUG


def test_ridge_zoo_preset_exists() -> None:
    config = from_spec("Ridge")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "ridge"


@pytest.mark.parametrize(
    ("predictor_class", "attributes"),
    [
        (ElasticNetPredictor, {"alpha": 1.0}),
        (LassoPredictor, {"alpha": 1.0, "max_iter": 10000, "tol": 1e-3, "selection": "random"}),
        (RidgePredictor, {}),
        (RandomForestPredictor, {"max_depth": 5, "criterion": "squared_error", "random_state": 42}),
        (SVRPredictor, {"C": 0.001, "epsilon": 0.001, "max_iter": 500, "kernel": "rbf"}),
        (
            GradientBoostingPredictor,
            {"max_iter": 100, "random_state": 42, "validation_fraction": 0.1, "n_iter_no_change": 10, "max_depth": 5},
        ),
        (AdaBoostPredictor, {"learning_rate": 1.0}),
        (KNNPredictor, {}),
    ],
)
def test_every_estimator_is_built_from_declared_defaults_matching_1_6_1(
    predictor_class: type[Predictor], attributes: dict[str, object]
) -> None:
    estimator = predictor_class()._make_estimator()

    for name, value in attributes.items():
        assert getattr(estimator, name) == value


def test_declared_spaces_agree_with_the_defaults() -> None:
    space = ElasticNetPredictor.get_hyperparameter_space()
    assert ElasticNetPredictor.get_default_hyperparameters() == {"alpha": 1.0, "l1_ratio": 0.0}
    assert (space["alpha"]["low"], space["alpha"]["high"], space["alpha"]["log"]) == (1e-4, 10.0, True)
    assert RandomForestPredictor.get_hyperparameter_space()["max_depth"]["default"] == 5
    assert AdaBoostPredictor.non_tunable_hyperparameters["learning_rate"] == 1.0
    assert AdaBoostPredictor()._make_estimator().estimator.max_depth == 4
    ada_space = AdaBoostPredictor.get_hyperparameter_space()
    assert (ada_space["min_samples_split"]["default"], ada_space["min_samples_leaf"]["default"]) == (2, 1)
    for predictor in (RandomForestPredictor, GradientBoostingPredictor):
        depth = predictor.get_hyperparameter_space()["max_depth"]
        assert all(depth["low"] <= value <= depth["high"] for value in (5, 10, 30))


def test_overridden_hyperparameters_reach_the_estimators() -> None:
    assert RandomForestPredictor({"random_state": 7})._make_estimator().random_state == 7
    assert RandomForestPredictor({"random_state": None})._make_estimator().random_state is None
    assert RandomForestPredictor({"max_depth": None})._make_estimator().max_depth is None
    assert GradientBoostingPredictor({"max_iter": 7})._make_estimator().max_iter == 7
    estimator = AdaBoostPredictor({"min_samples_split": 6, "min_samples_leaf": 3})._make_estimator()
    assert (estimator.estimator.min_samples_split, estimator.estimator.min_samples_leaf) == (6, 3)


def test_seeded_forest_and_boosting_fits_are_reproducible_by_default() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 5)), rng.normal(size=60)
    for predictor, params in (
        (RandomForestPredictor, {"n_estimators": 5, "n_jobs": 1}),
        (GradientBoostingPredictor, {"max_iter": 5}),
    ):
        first = predictor(params)._make_estimator().fit(x, y).predict(x)
        second = predictor(params)._make_estimator().fit(x, y).predict(x)
        np.testing.assert_array_equal(first, second)
    assert isinstance(ElasticNetPredictor()._make_estimator(), Ridge)
    assert SingleDrugElasticNetPredictor()._make_estimator().l1_ratio == 0.2


@pytest.mark.parametrize("zoo_name", ["ElasticNet", "SingleDrugElasticNet"])
def test_elasticnet_zoo_presets_do_not_override_alpha(zoo_name: str) -> None:
    resolved = ModelSpec.from_name(zoo_name).resolve()

    assert resolved.predictor_values()["alpha"] == 1.0


@pytest.mark.slow
def test_untuned_zoo_elasticnet_does_not_predict_a_constant_on_the_synthetic_dataset() -> None:
    import drevalpy

    dataset = drevalpy.synthetic()
    fold = drevalpy.data.split(dataset, "LPO", n_splits=2)[0]
    model = DRPModel(ModelSpec.from_name("ElasticNet"))

    model.train(dataset, fold.train)
    predictions = model.predict(dataset, fold.test)

    assert np.nanstd(predictions) > 0.05


def test_single_drug_knn_builds_scaler_mrmr_and_capped_knn() -> None:
    pipeline = SingleDrugKNNPredictor()._make_estimator()

    assert [name for name, _ in pipeline.steps] == ["scale", "mrmr", "knn"]
    assert isinstance(pipeline.named_steps["mrmr"], MRMRSelector)
    assert pipeline.named_steps["mrmr"].k == 20
    assert isinstance(pipeline.named_steps["knn"], NeighborCappedKNN)
    assert (pipeline.named_steps["knn"].n_neighbors, pipeline.named_steps["knn"].weights) == (3, "distance")
    for mrmr_k in (0, None):
        skipped = SingleDrugKNNPredictor({"mrmr_k": mrmr_k})._make_estimator()
        assert skipped.named_steps["mrmr"] == "passthrough"
    space = SingleDrugKNNPredictor.get_hyperparameter_space()
    assert (space["n_neighbors"]["high"], space["n_neighbors"]["default"]) == (20, 3)
    assert "mrmr_k" not in space
    assert SingleDrugKNNPredictor.non_tunable_hyperparameters["mrmr_k"] == 20


def test_single_drug_knn_standardizes_and_selects_per_drug() -> None:
    rng = np.random.RandomState(0)
    ids = [f"cl{i}" for i in range(6)]
    expression = rng.normal(size=(6, 4))
    source = MockFeatureSource(features={i: {"gene_expression": row} for i, row in zip(ids, expression, strict=True)})
    # d1 has six pairs and responds to gene 2; d2 has two pairs, fewer than n_neighbors.
    response = ResponseBatch(
        response=np.concatenate([expression[:, 2], [1.0, 2.0]]),
        sample_ids=np.array([*ids, "cl0", "cl1"]),
        drug_ids=np.array(["d1"] * 6 + ["d2"] * 2),
    )
    config = from_spec("scaledGeneExpression:identity:singleDrugKnn", hyperparameters={"mrmr_k": 2, "n_neighbors": 20})
    stack = _ModelPipeline(config)
    stack.reset()
    stack._fit_featurizers_and_predictor(response, source)

    predictor = stack._predictor
    assert isinstance(predictor, SingleDrugKNNPredictor)
    first, second = predictor._estimators["d1"], predictor._estimators["d2"]
    assert first.named_steps["mrmr"].selected_[0] == 2
    assert first.named_steps["knn"].n_neighbors == 6
    assert second.named_steps["knn"].n_neighbors == 2
    assert not np.allclose(first.named_steps["scale"].mean_, second.named_steps["scale"].mean_)
    query = ResponseBatch(response=None, sample_ids=np.array(["cl0", "cl0"]), drug_ids=np.array(["d1", "d2"]))
    assert np.isfinite(predictor.predict(stack._batch(query, source, None))).all()


def test_boosted_trees_shared_tables_are_consistent() -> None:
    assert set(SHARED_DEFAULTS) - {"random_state"} <= set(SHARED_SPACE)
    assert "log" not in SHARED_SPACE["learning_rate"]
    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in SHARED_SPACE.values())
    assert issubclass(BoostedTreesPredictor, SklearnTabularPredictor)


def test_estimator_params_resolve_defaults_overrides_and_coercion() -> None:
    assert _Bare()._estimator_params() == SHARED_DEFAULTS
    tweaked = _Tweaked()._estimator_params()
    assert tweaked["subsample"] == pytest.approx(0.8)
    assert tweaked["num_leaves"] == 63
    assert SHARED_DEFAULTS["subsample"] == pytest.approx(1.0)
    assert "num_leaves" not in _Bare(hyperparameters={"num_leaves": 5})._estimator_params()
    params = _Bare(hyperparameters={"n_estimators": "7", "learning_rate": "0.05"})._estimator_params()
    assert params["n_estimators"] == 7
    assert isinstance(params["n_estimators"], int)
    assert params["learning_rate"] == pytest.approx(0.05)


def test_boosted_hyperparameter_space_exposes_declared_names_merged_with_overrides() -> None:
    assert _Bare.get_hyperparameter_space() == {}
    space = _Tweaked.get_hyperparameter_space()
    assert set(space) == {"max_depth", "num_leaves"}
    assert space["max_depth"] == {**SHARED_SPACE["max_depth"], "high": 8}
    assert space["num_leaves"] == SHARED_SPACE["num_leaves"]
    assert SHARED_SPACE["max_depth"]["high"] == 12


@pytest.mark.parametrize(
    ("predictor_class", "keys"),
    [
        (
            XGBoostPredictor,
            {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"},
        ),
        (
            LightGBMPredictor,
            {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "num_leaves",
                "subsample",
                "colsample_bytree",
                "reg_alpha",
                "reg_lambda",
            },
        ),
    ],
    ids=["xgboost", "lightgbm"],
)
def test_boosted_predictors_tune_the_1_6_1_grid_keys_with_bounded_specs(
    predictor_class: type[BoostedTreesPredictor], keys: set[str]
) -> None:
    space = predictor_class.get_hyperparameter_space()

    assert issubclass(predictor_class, BoostedTreesPredictor)
    assert issubclass(predictor_class, SklearnTabularPredictor)
    assert set(space) == keys
    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in space.values())


def test_xgboost_estimator_params_keep_the_documented_defaults_and_forward_overrides() -> None:
    params = XGBoostPredictor()._estimator_params()

    assert (params["n_estimators"], params["max_depth"], params["random_state"]) == (100, 10, 42)
    assert params["learning_rate"] == pytest.approx(0.1)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.6)
    assert params["reg_alpha"] == pytest.approx(0.0)
    assert params["reg_lambda"] == pytest.approx(0.1)
    assert "num_leaves" not in params
    assert "log" not in XGBoostPredictor.get_hyperparameter_space()["learning_rate"]
    pytest.importorskip("xgboost")
    estimator = XGBoostPredictor(hyperparameters={"n_estimators": 7, "max_depth": 3})._make_estimator()
    assert (estimator.n_estimators, estimator.max_depth) == (7, 3)


def test_xgboost_load_applies_thread_defaults_before_restore() -> None:
    pytest.importorskip("xgboost")
    predictor = XGBoostPredictor(hyperparameters={"n_estimators": 5})
    predictor.fit(neural_batch())
    state = predictor.get_state()

    with patch(
        "drevalpy.components.predictors.xgboost_pred._set_xgboost_thread_defaults",
        wraps=_set_xgboost_thread_defaults,
    ) as thread_defaults:
        restored = XGBoostPredictor()
        restored.set_state(state)
        thread_defaults.assert_called_once()


def test_lightgbm_registry_name_contracts_and_class_hierarchy() -> None:
    cls = get_predictor("lightgbm")

    assert cls is LightGBMPredictor
    assert cls.sample_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX


def test_lightgbm_estimator_is_unfitted_with_defaults_and_forwarded_hyperparameters() -> None:
    estimator = LightGBMPredictor(hyperparameters=TINY_LIGHTGBM)._make_estimator()
    assert isinstance(estimator, lgb.LGBMRegressor)
    assert not hasattr(estimator, "booster_")

    default = LightGBMPredictor()._make_estimator()
    assert (default.n_estimators, default.max_depth, default.num_leaves, default.random_state) == (100, 6, 31, 42)
    assert default.learning_rate == pytest.approx(0.1)
    assert default.colsample_bytree == pytest.approx(0.6)
    assert default.get_params()["verbosity"] == -1
    space = LightGBMPredictor.get_hyperparameter_space()
    assert space["n_estimators"]["default"] == default.n_estimators
    assert space["num_leaves"]["default"] == default.num_leaves
    assert space["learning_rate"]["log"] is True

    forwarded = LightGBMPredictor(
        hyperparameters={"n_estimators": 7, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 5}
    )._make_estimator()
    assert (forwarded.n_estimators, forwarded.max_depth, forwarded.num_leaves) == (7, 3, 5)
    assert forwarded.learning_rate == pytest.approx(0.05)


def test_lightgbm_fit_predict_and_state_round_trip_reproduce_predictions() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_LIGHTGBM)
    assert predictor.is_fitted() is False

    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())

    assert predictor.is_fitted() is True
    assert expected.shape == (4,)
    assert np.isfinite(expected).all()
    restored = LightGBMPredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted() is True
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)
    with pytest.raises(PredictorStateError):
        LightGBMPredictor().set_state({"hyperparameters": dict(TINY_LIGHTGBM), "mode": "regression"})


def test_lightgbm_empty_training_matrix_leaves_it_unfitted() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_LIGHTGBM)

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))

    assert predictor.is_fitted() is False
    assert np.isnan(predictor._predict_matrix(np.zeros((2, 3)))).all()

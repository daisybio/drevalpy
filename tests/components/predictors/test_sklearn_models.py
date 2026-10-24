"""Tests for scikit-learn predictor component scope contracts."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
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
    SingleDrugRandomForestPredictor,
    SVRPredictor,
)
from drevalpy.models import ModelSpec
from drevalpy.models.config import ModelConfig, from_spec
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.types.enums.model_scope import ModelScope


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
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
    from sklearn.linear_model import Ridge

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
    import numpy as np

    import drevalpy
    from drevalpy.models import DRPModel

    dataset = drevalpy.synthetic()
    fold = drevalpy.data.split(dataset, "LPO", n_splits=2)[0]
    model = DRPModel(ModelSpec.from_name("ElasticNet"))

    model.train(dataset, fold.train)
    predictions = model.predict(dataset, fold.test)

    assert np.nanstd(predictions) > 0.05


def test_single_drug_knn_builds_scaler_mrmr_and_capped_knn() -> None:
    from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN
    from drevalpy.components.predictors.sklearn_models import SingleDrugKNNPredictor

    assert issubclass(SingleDrugKNNPredictor, SingleDrugSklearnPredictor)
    assert issubclass(SingleDrugKNNPredictor, KNNPredictor)
    assert SingleDrugKNNPredictor.scope is ModelScope.SINGLE_DRUG
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
    import numpy as np

    from drevalpy.components.predictors.sklearn_models import SingleDrugKNNPredictor
    from drevalpy.models._pipeline import _ModelPipeline
    from drevalpy.types.data.batch.response_batch import ResponseBatch
    from tests.conftest import MockFeatureSource

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

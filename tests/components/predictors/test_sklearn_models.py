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


@pytest.mark.parametrize("predictor_class", [ElasticNetPredictor, RandomForestPredictor])
def test_multi_drug_sklearn_predictors_are_multi_drug(predictor_class: type[Predictor]) -> None:
    assert predictor_class.scope is ModelScope.MULTI_DRUG


@pytest.mark.parametrize(
    ("predictor_class", "shared_predictor_class"),
    [
        (SingleDrugElasticNetPredictor, ElasticNetPredictor),
        (SingleDrugRandomForestPredictor, RandomForestPredictor),
    ],
)
def test_single_drug_sklearn_predictors_route_by_identity(
    predictor_class: type[Predictor],
    shared_predictor_class: type[Predictor],
) -> None:
    assert issubclass(predictor_class, shared_predictor_class)
    assert issubclass(predictor_class, SingleDrugSklearnPredictor)
    assert predictor_class.scope is ModelScope.SINGLE_DRUG


def test_ridge_zoo_preset_exists() -> None:
    config = from_spec("Ridge")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "ridge"


@pytest.mark.parametrize(
    "predictor_class",
    [
        ElasticNetPredictor,
        LassoPredictor,
        RidgePredictor,
        RandomForestPredictor,
        SVRPredictor,
        GradientBoostingPredictor,
        AdaBoostPredictor,
        KNNPredictor,
    ],
)
def test_every_estimator_is_built_from_declared_defaults_alone(predictor_class: type[Predictor]) -> None:
    """``_make_estimator`` indexes its keys, so a key without a declared default would raise ``KeyError``."""
    assert predictor_class()._make_estimator() is not None


def test_random_forest_defaults_follow_the_declared_space() -> None:
    estimator = RandomForestPredictor()._make_estimator()

    assert estimator.max_depth == RandomForestPredictor.get_hyperparameter_space()["max_depth"]["default"] == 5
    assert estimator.criterion == "squared_error"


def test_random_forest_is_seeded_with_42_unless_told_otherwise() -> None:
    assert RandomForestPredictor()._make_estimator().random_state == 42
    assert RandomForestPredictor({"random_state": 7})._make_estimator().random_state == 7
    assert RandomForestPredictor({"random_state": None})._make_estimator().random_state is None


def test_random_forest_fits_are_reproducible_by_default() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 5)), rng.normal(size=40)

    first = RandomForestPredictor({"n_estimators": 5, "n_jobs": 1})._make_estimator().fit(x, y).predict(x)
    second = RandomForestPredictor({"n_estimators": 5, "n_jobs": 1})._make_estimator().fit(x, y).predict(x)

    np.testing.assert_array_equal(first, second)


def test_random_forest_accepts_an_unbounded_depth() -> None:
    assert RandomForestPredictor({"max_depth": None})._make_estimator().max_depth is None


def test_adaboost_learning_rate_is_a_declared_non_tunable_default() -> None:
    assert AdaBoostPredictor.non_tunable_hyperparameters["learning_rate"] == 1.0
    assert AdaBoostPredictor()._make_estimator().learning_rate == 1.0


def test_gradient_boosting_iterations_follow_max_iter() -> None:
    assert GradientBoostingPredictor()._make_estimator().max_iter == 100
    assert GradientBoostingPredictor({"max_iter": 7})._make_estimator().max_iter == 7


def test_adaboost_default_depth_matches_space() -> None:
    predictor = AdaBoostPredictor()
    estimator = predictor._make_estimator()
    assert estimator.estimator.max_depth == 4


def test_elasticnet_defaults_follow_the_first_grid_values_of_1_6_1_with_an_unchanged_search_range() -> None:
    space = ElasticNetPredictor.get_hyperparameter_space()

    assert ElasticNetPredictor.get_default_hyperparameters() == {"alpha": 1.0, "l1_ratio": 0.0}
    assert (space["alpha"]["low"], space["alpha"]["high"], space["alpha"]["log"]) == (1e-4, 10.0, True)
    assert ElasticNetPredictor()._make_estimator().alpha == 1.0


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


@pytest.mark.parametrize("mrmr_k", [0, None])
def test_single_drug_knn_without_mrmr_k_skips_the_selection(mrmr_k: int | None) -> None:
    from drevalpy.components.predictors.sklearn_models import SingleDrugKNNPredictor

    pipeline = SingleDrugKNNPredictor({"mrmr_k": mrmr_k})._make_estimator()
    assert pipeline.named_steps["mrmr"] == "passthrough"


def test_single_drug_knn_space_reaches_twenty_neighbors_and_leaves_mrmr_k_untuned() -> None:
    from drevalpy.components.predictors.sklearn_models import SingleDrugKNNPredictor

    space = SingleDrugKNNPredictor.get_hyperparameter_space()
    assert (space["n_neighbors"]["high"], space["n_neighbors"]["default"]) == (20, 3)
    assert "mrmr_k" not in space
    assert SingleDrugKNNPredictor.non_tunable_hyperparameters["mrmr_k"] == 20


def test_single_drug_knn_standardizes_and_selects_per_drug() -> None:
    """Each drug gets its own scaler, its own genes and as many neighbours as it has pairs."""
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


def test_gradient_boosting_is_seeded_and_validates_on_a_tenth_as_in_1_6_1() -> None:
    estimator = GradientBoostingPredictor()._make_estimator()

    assert estimator.random_state == 42
    assert estimator.validation_fraction == 0.1
    assert estimator.n_iter_no_change == 10
    assert estimator.max_depth == 5


def test_gradient_boosting_fits_are_reproducible_by_default() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 4)), rng.normal(size=60)

    first = GradientBoostingPredictor({"max_iter": 5})._make_estimator().fit(x, y).predict(x)
    second = GradientBoostingPredictor({"max_iter": 5})._make_estimator().fit(x, y).predict(x)

    np.testing.assert_array_equal(first, second)


def test_svr_defaults_are_the_first_grid_values_of_1_6_1() -> None:
    estimator = SVRPredictor()._make_estimator()

    assert (estimator.C, estimator.epsilon, estimator.max_iter, estimator.kernel) == (0.001, 0.001, 500, "rbf")


def test_lasso_solver_settings_are_those_of_1_6_1() -> None:
    estimator = LassoPredictor()._make_estimator()

    assert (estimator.alpha, estimator.max_iter, estimator.tol, estimator.selection) == (1.0, 10000, 1e-3, "random")


def test_elasticnet_at_l1_ratio_zero_is_ridge_and_single_drug_uses_0_2() -> None:
    from sklearn.linear_model import Ridge

    assert isinstance(ElasticNetPredictor()._make_estimator(), Ridge)
    assert SingleDrugElasticNetPredictor()._make_estimator().l1_ratio == 0.2


def test_adaboost_split_and_leaf_sizes_are_tunable() -> None:
    space = AdaBoostPredictor.get_hyperparameter_space()

    assert space["min_samples_split"]["default"] == 2
    assert space["min_samples_leaf"]["default"] == 1
    estimator = AdaBoostPredictor({"min_samples_split": 6, "min_samples_leaf": 3})._make_estimator()
    assert (estimator.estimator.min_samples_split, estimator.estimator.min_samples_leaf) == (6, 3)


@pytest.mark.parametrize(
    ("name", "key", "values"),
    [
        ("RandomForest", "max_depth", {5, 10, 30}),
        ("GradientBoosting", "max_depth", {5, 10, 30}),
    ],
)
def test_tunable_depth_ranges_cover_the_1_6_1_grid(name: str, key: str, values: set[int]) -> None:
    predictor = {"RandomForest": RandomForestPredictor, "GradientBoosting": GradientBoostingPredictor}[name]
    spec = predictor.get_hyperparameter_space()[key]

    assert all(spec["low"] <= value <= spec["high"] for value in values)

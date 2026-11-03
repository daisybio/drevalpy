"""``DRPModel``: lifecycle states, training context, component construction and feature preparation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy._utils import accelerator
from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.errors import DrEvalError, InvalidInputError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.drp_model import ModelState, _build_components, validate_fit_scopes
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types import PairMask
from drevalpy.types.enums.accelerator import Accelerator
from tests.models.synthetic_fixtures import (
    lso_fold,
    small_data_spec,
    synthetic_model_dataset_gene_expression_fingerprints,
    synthetic_model_dataset_identity,
)


def test_constructor_and_properties() -> None:
    with pytest.raises(TypeError, match="ModelSpec"):
        DRPModel("ElasticNet")  # type: ignore[arg-type]

    model = DRPModel(small_data_spec("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5})

    assert model.name == "ElasticNet"
    assert model.hyperparameters["predictor.elasticNet.alpha"] == 0.1
    assert model.spec.hyperparameters["predictor.elasticNet.l1_ratio"] == 0.5
    values = model.hyperparameters
    values["predictor.elasticNet.alpha"] = 99
    assert model.hyperparameters["predictor.elasticNet.alpha"] == 0.1


def test_check_values_false_takes_values_outside_the_space_but_not_unknown_keys() -> None:
    spec = small_data_spec("RandomForest")

    with pytest.raises(ValueError, match="is outside its search space"):
        DRPModel(spec, {"n_estimators": 5})
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        DRPModel(spec, {"nope": 1}, check_values=False)
    model = DRPModel(spec, {"n_estimators": 5}, check_values=False)

    assert model.hyperparameters["predictor.randomForest.n_estimators"] == 5


def test_train_predict_and_load_preserve_original_response_units(tmp_path) -> None:
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    fold = lso_fold()
    model = DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler())

    model.train(dataset, fold.train, model_checkpoint_dir=tmp_path)
    predictions = model.predict(dataset, fold.test)
    model.save(tmp_path / "model")
    restored = DRPModel.load(tmp_path / "model")

    assert model.state is ModelState.FITTED
    assert predictions.shape == (2,)
    np.testing.assert_allclose(restored.predict(dataset, fold.test), predictions)


def test_fit_scope_validation() -> None:
    dataset = synthetic_model_dataset_identity()
    scope = PairMask.from_pairs(np.array([[0, 0]]), shape=dataset.response_matrix.shape)

    with pytest.raises(ValueError, match="response-disjoint"):
        validate_fit_scopes(dataset, scope, scope)
    with pytest.raises(TypeError, match="Dataset"):
        validate_fit_scopes(object(), PairMask(np.zeros((2, 2), dtype=bool)), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        validate_fit_scopes(dataset, PairMask(np.zeros((1, 1), dtype=bool)), None)


def test_train_without_a_checkpoint_dir_uses_a_temporary_one_and_removes_it(tmp_path, monkeypatch) -> None:
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    fold = lso_fold()
    model = DRPModel(small_data_spec("ElasticNet"))
    seen = []
    original_fit = model._fit

    def spy(*args, training_context, **kwargs):
        seen.append(training_context.checkpoint_dir)
        assert training_context.checkpoint_dir.is_dir()
        return original_fit(*args, training_context=training_context, **kwargs)

    monkeypatch.setattr(model, "_fit", spy)
    monkeypatch.chdir(tmp_path)

    model.train(dataset, fold.train)

    assert len(seen) == 1
    assert not seen[0].exists()
    assert not (tmp_path / "checkpoints").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("n_jobs", "expected"), [(None, 1), (3, 3)])
def test_train_puts_n_jobs_in_the_training_context(tmp_path, monkeypatch, n_jobs, expected) -> None:
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    model = DRPModel(small_data_spec("ElasticNet"))
    seen = []
    original_fit = model._fit

    def spy(*args, training_context, **kwargs):
        seen.append(training_context.n_jobs)
        return original_fit(*args, training_context=training_context, **kwargs)

    monkeypatch.setattr(model, "_fit", spy)

    extra = {} if n_jobs is None else {"n_jobs": n_jobs}
    model.train(dataset, lso_fold().train, model_checkpoint_dir=tmp_path, **extra)

    assert seen == [expected]


def test_train_rejects_an_unusable_n_jobs() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))

    with pytest.raises(InvalidInputError, match="n_jobs"):
        model.train(synthetic_model_dataset_gene_expression_fingerprints(), lso_fold().train, n_jobs=0)


def test_model_starts_untrained_and_predict_rejects_it() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    dataset = synthetic_model_dataset_gene_expression_fingerprints()

    assert model.state is ModelState.UNTRAINED
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(dataset, lso_fold().test)


def test_empty_fit_returns_aligned_nan_predictions() -> None:
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    dataset = synthetic_model_dataset_identity()
    empty = PairMask(np.zeros(dataset.response_matrix.shape, dtype=bool))

    model.train(dataset, empty)

    assert model.state is ModelState.EMPTY
    assert np.isnan(model.predict(dataset, lso_fold().test)).all()


def test_retraining_resets_previous_fitted_state() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    fold = lso_fold()

    model.train(dataset, fold.train)
    assert model.state is ModelState.FITTED
    model.train(dataset, PairMask(np.zeros(dataset.response_matrix.shape, dtype=bool)))

    assert model.state is ModelState.EMPTY
    assert np.isnan(model.predict(dataset, fold.test)).all()


def _with_gene_columns(dataset, columns: slice):
    gene_expression = dataset.mdata.mod["gene_expression"]
    dataset.mdata.mod["gene_expression"] = gene_expression[:, columns].copy()
    dataset.mdata.update()
    return dataset


def test_predictions_do_not_depend_on_the_gene_order_of_the_dataset(tmp_path) -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(synthetic_model_dataset_gene_expression_fingerprints(), lso_fold().train)
    model.save(tmp_path / "model")
    restored = DRPModel.load(tmp_path / "model")
    reordered = _with_gene_columns(synthetic_model_dataset_gene_expression_fingerprints(), slice(None, None, -1))
    test = lso_fold().test

    expected = model.predict(synthetic_model_dataset_gene_expression_fingerprints(), test)

    np.testing.assert_allclose(model.predict(reordered, test), expected)
    np.testing.assert_allclose(restored.predict(reordered, test), expected)


def test_predicting_on_a_dataset_without_a_fitted_gene_raises(tmp_path) -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(synthetic_model_dataset_gene_expression_fingerprints(), lso_fold().train)
    model.save(tmp_path / "model")
    missing_gene = _with_gene_columns(synthetic_model_dataset_gene_expression_fingerprints(), slice(1, None))

    for fitted in (model, DRPModel.load(tmp_path / "model")):
        with pytest.raises(FeatureMismatchError, match="gene_expression.*missing 1 of the 3 features.*'gene0'"):
            fitted.predict(missing_gene, lso_fold().test)


@pytest.mark.parametrize("failure", ["validation", "construction", "fit"])
def test_entered_training_failure_invalidates_previous_model(monkeypatch, failure) -> None:
    import drevalpy.models.drp_model as drp_model_module

    dataset = synthetic_model_dataset_identity()
    scope = lso_fold().train
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, scope)

    def fail(*args, **kwargs):
        raise ValueError("deliberate failure")

    if failure == "construction":
        monkeypatch.setattr(drp_model_module, "_build_components", fail)
    elif failure == "fit":
        monkeypatch.setattr(type(model._predictor), "fit", fail)
    else:
        scope = PairMask(np.zeros((1, 1), dtype=bool))
    with pytest.raises(ValueError):
        model.train(dataset, scope)
    assert model.state is ModelState.UNTRAINED
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(dataset, lso_fold().test)


def test_argument_binding_failure_preserves_previous_fit() -> None:
    dataset = synthetic_model_dataset_identity()
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, lso_fold().train)
    expected = model.predict(dataset, lso_fold().test)
    with pytest.raises(TypeError):
        model.train(dataset, unexpected=True)
    assert model.state is ModelState.FITTED
    np.testing.assert_array_equal(model.predict(dataset, lso_fold().test), expected)


def test_components_construct_once_per_training_attempt(monkeypatch) -> None:
    import drevalpy.models.drp_model as drp_model_module

    calls = []
    build = drp_model_module._build_components

    def record(spec):
        components = build(spec)
        calls.append(components)
        return components

    monkeypatch.setattr(drp_model_module, "_build_components", record)
    model = DRPModel(small_data_spec("ElasticNet"))
    assert calls == []
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    model.train(dataset, lso_fold().train)
    model.train(dataset, lso_fold().train)
    assert len(calls) == 2
    assert all(first is not second for first, second in zip(calls[0], calls[1], strict=True))


@pytest.mark.parametrize("trained", [False, True])
def test_unfitted_or_empty_model_cannot_be_saved(tmp_path, trained) -> None:
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    if trained:
        dataset = synthetic_model_dataset_identity()
        model.train(dataset, PairMask(np.zeros((2, 2), dtype=bool)))
    with pytest.raises(RuntimeError, match="must be trained first"):
        model.save(tmp_path / "model.zip")


def test_empty_prediction_does_not_access_dataset_features(monkeypatch) -> None:
    dataset = synthetic_model_dataset_identity()
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, PairMask(np.zeros((2, 2), dtype=bool)))

    def fail(*args, **kwargs):
        raise AssertionError("Empty model must not read features")

    monkeypatch.setattr(model, "_feature_sources", fail)
    scope = PairMask(np.ones((2, 2), dtype=bool)).shuffled(7)
    values = model.predict(dataset, scope)
    assert values.shape == (4,)
    assert np.isnan(values).all()


def test_prediction_retains_unobserved_targets_and_all_invalid_feature_rows() -> None:
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(dataset, lso_fold().train)
    dataset.mdata.mod["response"].X[1, 0] = np.nan
    requested = lso_fold().test.shuffled(1)
    assert model.predict(dataset, requested).shape == (2,)
    assert np.isfinite(model.predict(dataset, requested)).all()
    dataset.mdata.mod["gene_expression"].X[1, :] = np.nan
    values = model.predict(dataset, requested)
    assert values.shape == (2,)
    assert np.isnan(values).all()


def test_each_side_checks_and_computes_features_once_per_operation(monkeypatch) -> None:
    spec = small_data_spec("ElasticNet")
    calls: list[tuple[str, str]] = []
    for featurizer in _build_components(spec)[:2]:
        cls = type(featurizer)
        for method in ("valid_mask", "transform"):
            original = getattr(cls, method)

            def record(self, source, ids, _original=original, _method=method):
                calls.append((type(self).__name__, _method))
                return _original(self, source, ids)

            monkeypatch.setattr(cls, method, record)
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    model = DRPModel(spec)
    model.train(dataset, lso_fold().train)
    assert sorted(method for _, method in calls) == ["transform", "transform", "valid_mask", "valid_mask"]
    calls.clear()
    model.predict(dataset, lso_fold().test)
    assert sorted(method for _, method in calls) == ["transform", "transform", "valid_mask", "valid_mask"]


def test_mean_effects_falls_back_to_sample_and_drug_effects_without_tissue_metadata() -> None:
    dataset = synthetic_model_dataset_identity()
    assert "tissue" not in dataset.sample_meta.columns
    fold = lso_fold()
    model = DRPModel(ModelSpec.from_name("NaiveMeanEffectsPredictor"))

    model.train(dataset, fold.train)
    preds = model.predict(dataset, fold.test)

    assert model.state is ModelState.FITTED
    assert np.isfinite(preds).all()


def test_build_components_creates_independent_featurizers_and_predictor() -> None:
    spec = ModelSpec.from_name("ElasticNet").with_hyperparameters({"alpha": 0.25, "n_bits": 512})
    first = _build_components(spec)
    second = _build_components(spec)

    assert all(component is not None for component in first)
    assert all(a is not b for a, b in zip(first, second, strict=True))
    assert first[2]._hyperparameters["alpha"] == 0.25
    assert first[1]._n_bits == 512
    mode = first[2]._hyperparameters["prediction_mode"]
    assert mode == "regression"
    assert type(mode) is str


def test_a_model_that_requires_an_accelerator_fails_fast_without_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(get_predictor("elasticNet"), "accelerator", Accelerator.REQUIRED)
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    fold = lso_fold()
    model = DRPModel(small_data_spec("ElasticNet"))

    monkeypatch.setattr(accelerator, "accelerator_available", lambda: False)
    with pytest.raises(DrEvalError, match="'ElasticNet' requires an accelerator"):
        model.train(dataset, fold.train, model_checkpoint_dir=tmp_path)

    monkeypatch.setattr(accelerator, "accelerator_available", lambda: True)
    model.train(dataset, fold.train, model_checkpoint_dir=tmp_path)
    monkeypatch.setattr(accelerator, "accelerator_available", lambda: False)
    with pytest.raises(DrEvalError, match="'ElasticNet' requires an accelerator"):
        model.predict(dataset, fold.test)


class TestResponseLayers:
    """The response layers a predictor declares reach its fit, untransformed, and never its predict."""

    @pytest.fixture()
    def model(self, monkeypatch) -> tuple[DRPModel, list]:
        from drevalpy.components.predictors.abstract.base import Predictor

        monkeypatch.setattr(Predictor, "response_layers", ("regulation",))
        seen = []
        for method in ("fit", "predict"):
            original = getattr(Predictor, method)

            def spy(self, batch, _original=original, _method=method):
                seen.append((_method, batch))
                return _original(self, batch)

            monkeypatch.setattr(Predictor, method, spy)
        return DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler()), seen

    def test_fit_gets_the_layer_untransformed_and_predict_gets_none(self, tmp_path, model) -> None:
        dataset = synthetic_model_dataset_gene_expression_fingerprints()
        dataset.response.layers["regulation"] = np.array([[-1.0, 1.0], [0.0, np.nan]], dtype=np.float32)
        model, seen = model

        model.train(dataset, lso_fold().train, model_checkpoint_dir=tmp_path)
        model.predict(dataset, lso_fold().test)

        (_, fitted), (_, predicted) = seen
        assert fitted.sample_ids.tolist() == ["cl1", "cl1"]
        assert fitted.response_layers["regulation"].tolist() == [-1.0, 1.0]
        assert not np.allclose(fitted.response, [1.0, 2.0])
        assert dict(predicted.response_layers) == {}

    def test_a_dataset_without_the_layer_trains_without_it(self, tmp_path, model) -> None:
        model, seen = model

        model.train(
            synthetic_model_dataset_gene_expression_fingerprints(), lso_fold().train, model_checkpoint_dir=tmp_path
        )

        assert dict(seen[0][1].response_layers) == {}

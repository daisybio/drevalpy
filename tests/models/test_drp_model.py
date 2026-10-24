"""``DRPModel`` and its pipeline: lifecycle states, training context, component construction and feature preparation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.errors import InvalidInputError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models._components import build_components
from drevalpy.models._feature_preparation import transform_entities, usable_responses
from drevalpy.models._pipeline import PipelineState
from drevalpy.models.config import from_spec
from drevalpy.models.tuning.search_space import resolve_model_config
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)


def test_constructor_and_properties() -> None:
    with pytest.raises(TypeError, match="ModelSpec"):
        DRPModel("ElasticNet")  # type: ignore[arg-type]

    model = DRPModel(small_data_spec("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5})

    assert model.name == "ElasticNet"
    assert model.config.predictor_values()["alpha"] == 0.1
    values = model.hyperparameters
    values["alpha"] = 99
    assert model.hyperparameters["alpha"] == 0.1


def test_train_predict_and_load_preserve_original_response_units(tmp_path) -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler())

    model.train(dataset, split.train, model_checkpoint_dir=tmp_path)
    predictions = model.predict(dataset, split.test)
    model.save(tmp_path / "model")
    restored = DRPModel.load(tmp_path / "model")

    assert model._pipeline.state is PipelineState.FITTED
    assert predictions.shape == (2,)
    np.testing.assert_allclose(restored.predict(dataset, split.test), predictions)


def test_train_without_a_checkpoint_dir_uses_a_temporary_one_and_removes_it(tmp_path, monkeypatch) -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(small_data_spec("ElasticNet"))
    seen = []
    original_fit = model._pipeline.fit

    def spy(*args, training_context, **kwargs):
        seen.append(training_context.checkpoint_dir)
        assert training_context.checkpoint_dir.is_dir()
        return original_fit(*args, training_context=training_context, **kwargs)

    monkeypatch.setattr(model._pipeline, "fit", spy)
    monkeypatch.chdir(tmp_path)

    model.train(dataset, split.train)

    assert len(seen) == 1
    assert not seen[0].exists()
    assert not (tmp_path / "checkpoints").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("n_jobs", "expected"), [(None, 1), (3, 3)])
def test_train_puts_n_jobs_in_the_training_context(tmp_path, monkeypatch, n_jobs, expected) -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    model = DRPModel(small_data_spec("ElasticNet"))
    seen = []
    original_fit = model._pipeline.fit

    def spy(*args, training_context, **kwargs):
        seen.append(training_context.n_jobs)
        return original_fit(*args, training_context=training_context, **kwargs)

    monkeypatch.setattr(model._pipeline, "fit", spy)

    extra = {} if n_jobs is None else {"n_jobs": n_jobs}
    model.train(dataset, lso_split_masks().train, model_checkpoint_dir=tmp_path, **extra)

    assert seen == [expected]


def test_train_rejects_an_unusable_n_jobs() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))

    with pytest.raises(InvalidInputError, match="n_jobs"):
        model.train(synthetic_mudataset_gene_expression_fingerprints(), lso_split_masks().train, n_jobs=0)


def test_model_starts_untrained_and_predict_rejects_it() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    dataset = synthetic_mudataset_gene_expression_fingerprints()

    assert model._pipeline.state is PipelineState.UNTRAINED
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(dataset, lso_split_masks().test)


def test_empty_fit_returns_aligned_nan_predictions() -> None:
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    dataset = synthetic_mudataset_identity()
    empty = SplitMask(np.zeros(dataset.response_matrix.shape, dtype=bool))

    model.train(dataset, empty)

    assert model._pipeline.state is PipelineState.EMPTY
    assert np.isnan(model.predict(dataset, lso_split_masks().test)).all()


def test_retraining_resets_previous_fitted_state() -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    masks = lso_split_masks()

    model.train(dataset, masks.train)
    assert model._pipeline.state is PipelineState.FITTED
    model.train(dataset, SplitMask(np.zeros(dataset.response_matrix.shape, dtype=bool)))

    assert model._pipeline.state is PipelineState.EMPTY
    assert np.isnan(model.predict(dataset, masks.test)).all()


def _with_gene_columns(dataset, columns: slice):
    gene_expression = dataset.mdata.mod["gene_expression"]
    dataset.mdata.mod["gene_expression"] = gene_expression[:, columns].copy()
    dataset.mdata.update()
    return dataset


def test_predictions_do_not_depend_on_the_gene_order_of_the_dataset(tmp_path) -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(synthetic_mudataset_gene_expression_fingerprints(), lso_split_masks().train)
    model.save(tmp_path / "model")
    restored = DRPModel.load(tmp_path / "model")
    reordered = _with_gene_columns(synthetic_mudataset_gene_expression_fingerprints(), slice(None, None, -1))
    test = lso_split_masks().test

    expected = model.predict(synthetic_mudataset_gene_expression_fingerprints(), test)

    np.testing.assert_allclose(model.predict(reordered, test), expected)
    np.testing.assert_allclose(restored.predict(reordered, test), expected)


def test_predicting_on_a_dataset_without_a_fitted_gene_raises(tmp_path) -> None:
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(synthetic_mudataset_gene_expression_fingerprints(), lso_split_masks().train)
    model.save(tmp_path / "model")
    missing_gene = _with_gene_columns(synthetic_mudataset_gene_expression_fingerprints(), slice(1, None))

    for fitted in (model, DRPModel.load(tmp_path / "model")):
        with pytest.raises(FeatureMismatchError, match="gene_expression.*missing 1 of the 3 features.*'gene0'"):
            fitted.predict(missing_gene, lso_split_masks().test)


@pytest.mark.parametrize("failure", ["validation", "construction", "fit"])
def test_entered_training_failure_invalidates_previous_model(monkeypatch, failure) -> None:
    import drevalpy.models._pipeline as pipeline_module

    dataset = synthetic_mudataset_identity()
    scope = lso_split_masks().train
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, scope)

    def fail(*args, **kwargs):
        raise ValueError("deliberate failure")

    if failure == "construction":
        monkeypatch.setattr(pipeline_module, "build_components", fail)
    elif failure == "fit":
        monkeypatch.setattr(type(model._pipeline._predictor), "fit", fail)
    else:
        scope = SplitMask(np.zeros((1, 1), dtype=bool))
    with pytest.raises(ValueError):
        model.train(dataset, scope)
    assert model._pipeline.state is PipelineState.UNTRAINED
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(dataset, lso_split_masks().test)


def test_argument_binding_failure_preserves_previous_fit() -> None:
    dataset = synthetic_mudataset_identity()
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, lso_split_masks().train)
    expected = model.predict(dataset, lso_split_masks().test)
    with pytest.raises(TypeError):
        model.train(dataset, unexpected=True)
    assert model._pipeline.state is PipelineState.FITTED
    np.testing.assert_array_equal(model.predict(dataset, lso_split_masks().test), expected)


def test_components_construct_once_per_training_attempt(monkeypatch) -> None:
    import drevalpy.models._pipeline as pipeline_module

    calls = []
    build = pipeline_module.build_components

    def record(config):
        components = build(config)
        calls.append(components)
        return components

    monkeypatch.setattr(pipeline_module, "build_components", record)
    model = DRPModel(small_data_spec("ElasticNet"))
    assert calls == []
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    model.train(dataset, lso_split_masks().train)
    model.train(dataset, lso_split_masks().train)
    assert len(calls) == 2
    assert all(first is not second for first, second in zip(calls[0], calls[1], strict=True))


@pytest.mark.parametrize("trained", [False, True])
def test_unfitted_or_empty_pipeline_cannot_be_saved(tmp_path, trained) -> None:
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    if trained:
        dataset = synthetic_mudataset_identity()
        model.train(dataset, SplitMask(np.zeros((2, 2), dtype=bool)))
    with pytest.raises(RuntimeError, match="FITTED"):
        model.save(tmp_path / "model.zip")


def test_empty_prediction_does_not_access_dataset_features(monkeypatch) -> None:
    dataset = synthetic_mudataset_identity()
    model = DRPModel(ModelSpec.from_name("NaivePredictor"))
    model.train(dataset, SplitMask(np.zeros((2, 2), dtype=bool)))

    def fail(*args, **kwargs):
        raise AssertionError("Empty pipeline must not read features")

    monkeypatch.setattr(model._pipeline, "_build_features_from_mudataset", fail)
    scope = SplitMask(np.ones((2, 2), dtype=bool)).shuffled(7)
    values = model.predict(dataset, scope)
    assert values.shape == (4,)
    assert np.isnan(values).all()


def test_prediction_retains_unobserved_targets_and_all_invalid_feature_rows() -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(dataset, lso_split_masks().train)
    dataset.mdata.mod["response"].X[1, 0] = np.nan
    requested = lso_split_masks().test.shuffled(1)
    assert model.predict(dataset, requested).shape == (2,)
    assert np.isfinite(model.predict(dataset, requested)).all()
    dataset.mdata.mod["gene_expression"].X[1, :] = np.nan
    values = model.predict(dataset, requested)
    assert values.shape == (2,)
    assert np.isnan(values).all()


def test_each_side_computes_features_once_per_operation(monkeypatch) -> None:
    from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer

    calls = []
    transform = DenseViewFeaturizer._transform_features

    def record(self, source, ids):
        calls.append(type(self).__name__)
        return transform(self, source, ids)

    monkeypatch.setattr(DenseViewFeaturizer, "_transform_features", record)
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    model = DRPModel(small_data_spec("ElasticNet"))
    model.train(dataset, lso_split_masks().train)
    assert len(calls) == 2
    calls.clear()
    model.predict(dataset, lso_split_masks().test)
    assert len(calls) == 2


def test_mean_effects_falls_back_to_sample_and_drug_effects_without_tissue_metadata() -> None:
    dataset = synthetic_mudataset_identity()
    assert "tissue" not in dataset.sample_meta.columns
    split = lso_split_masks()
    model = DRPModel(ModelSpec.from_name("NaiveMeanEffectsPredictor"))

    model.train(dataset, split.train)
    preds = model.predict(dataset, split.test)

    assert model._pipeline.state is PipelineState.FITTED
    assert np.isfinite(preds).all()


def test_build_components_creates_independent_featurizers_and_predictor() -> None:
    resolved = resolve_model_config(from_spec("ElasticNet"))
    first = build_components(resolved)
    second = build_components(resolved)

    assert all(component is not None for component in first)
    assert all(a is not b for a, b in zip(first, second, strict=True))
    mode = first[2]._hyperparameters["prediction_mode"]
    assert mode == "regression"
    assert type(mode) is str


def test_transform_without_a_featurizer_returns_empty_result() -> None:
    result = transform_entities(None, None, np.array(["a", "b"]))
    np.testing.assert_array_equal(result.entity_ids, ["a", "b"])
    assert result.matrix.shape == (2, 0)
    assert result.blocks == {}
    assert result.projection == ()


def test_configured_featurizer_requires_a_feature_source() -> None:
    with pytest.raises(ValueError, match="Feature source is required"):
        transform_entities(object(), None, np.array(["a"]))  # type: ignore[arg-type]


def test_usable_responses_removes_missing_targets() -> None:
    response = ResponseBatch(
        response=np.array([1.0, np.nan, 3.0]),
        sample_ids=np.array(["a", "a", "b"]),
        drug_ids=np.array(["x", "y", "x"]),
    )
    usable = usable_responses(response, None, None, None, None)
    np.testing.assert_array_equal(usable.response, [1.0, 3.0])
    np.testing.assert_array_equal(usable.sample_ids, ["a", "b"])

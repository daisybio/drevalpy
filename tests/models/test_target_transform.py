"""Tests for fitted target transformations used by the model pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from drevalpy.models._target_transform import apply_target_transform, fit_target_transform, validate_fit_scopes
from drevalpy.types import SplitMask
from tests.models.synthetic_fixtures import synthetic_mudataset_identity


class _CenteringTransformer(BaseEstimator):
    """Custom cloneable target transformer with a learned center and offset."""

    def __init__(self, offset=7.0):
        self.offset = offset

    def fit(self, values):
        self.mean_ = np.mean(values, axis=0)
        return self

    def transform(self, values):
        return values - self.mean_ + self.offset

    def inverse_transform(self, values):
        return values + self.mean_ - self.offset


def test_fit_clones_transformer_and_preserves_missing_values() -> None:
    prototype = StandardScaler()
    fitted = fit_target_transform(prototype, np.array([1.0, np.nan, 3.0]))

    assert fitted is not prototype
    transformed = apply_target_transform(fitted, np.array([1.0, np.nan, 3.0]))
    assert np.isnan(transformed[1])
    np.testing.assert_allclose(
        apply_target_transform(fitted, transformed, inverse=True), [1.0, np.nan, 3.0], equal_nan=True
    )


def test_none_transform_is_identity() -> None:
    values = np.array([1.0, np.nan, 2.0])
    np.testing.assert_array_equal(apply_target_transform(None, values), values)


def test_invalid_transformer_is_rejected() -> None:
    with pytest.raises(TypeError, match="fit, transform, and inverse_transform"):
        fit_target_transform(object(), np.array([1.0]))


def test_non_vector_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_target_transform(None, np.ones((2, 1)))


def test_fit_scope_validation_rejects_overlap() -> None:
    dataset = synthetic_mudataset_identity()
    scope = SplitMask.from_pairs(np.array([[0, 0]]), shape=dataset.response_matrix.shape)

    with pytest.raises(ValueError, match="response-disjoint"):
        validate_fit_scopes(dataset, scope, scope)


def test_fit_scope_validation_requires_dataset_and_matching_shape() -> None:
    with pytest.raises(TypeError, match="Dataset"):
        validate_fit_scopes(object(), SplitMask(np.zeros((2, 2), dtype=bool)), None)  # type: ignore[arg-type]

    dataset = synthetic_mudataset_identity()
    with pytest.raises(ValueError, match="shape"):
        validate_fit_scopes(dataset, SplitMask(np.zeros((1, 1), dtype=bool)), None)


def test_custom_transform_is_independent_and_round_trips_with_retraining(tmp_path) -> None:
    from drevalpy.models import DRPModel, ModelSpec
    from tests.models.synthetic_fixtures import lso_split_masks

    prototype = _CenteringTransformer(offset=11.0)
    dataset = synthetic_mudataset_identity()
    first = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=prototype)
    second = DRPModel(first.spec, response_transformation=prototype)
    scope = lso_split_masks().train
    first.train(dataset, scope)
    second.train(dataset, lso_split_masks().test)
    assert not hasattr(prototype, "mean_")
    assert first._pipeline._response_transform is not second._pipeline._response_transform
    np.testing.assert_allclose(first.predict(dataset, scope), 1.5)
    np.testing.assert_allclose(second.predict(dataset, scope), 3.5)
    first.save(tmp_path / "transformed")
    restored = DRPModel.load(tmp_path / "transformed")
    np.testing.assert_array_equal(restored.predict(dataset, scope), first.predict(dataset, scope))
    old_transform = restored._pipeline._response_transform
    restored.train(dataset, lso_split_masks().test)
    assert restored._pipeline._response_transform is not old_transform
    np.testing.assert_allclose(restored.predict(dataset, scope), 3.5)


def test_transform_fits_original_training_population_before_feature_filtering() -> None:
    from drevalpy.models import DRPModel, ModelSpec
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["gene_expression"].X[1, :] = np.nan
    model = DRPModel(ModelSpec.from_name("ElasticNet"), response_transformation=StandardScaler())
    model.train(dataset, SplitMask(np.ones((2, 2), dtype=bool)))
    assert model._pipeline._response_transform.mean_[0] == pytest.approx(2.5)


def test_stopping_targets_are_transformed_using_training_statistics(monkeypatch) -> None:
    from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
    from drevalpy.models import DRPModel, ModelSpec

    observed = []
    fit = NaiveMeanPredictor._fit

    def record(self, batch):
        observed.append(batch)
        return fit(self, batch)

    monkeypatch.setattr(NaiveMeanPredictor, "_fit", record)
    dataset = synthetic_mudataset_identity()
    model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=StandardScaler())
    training = SplitMask.from_pairs(np.array([[0, 0], [1, 0]]), (2, 2))
    stopping = SplitMask.from_pairs(np.array([[1, 1]]), (2, 2))
    model.train(dataset, training, early_stopping_scope=stopping)
    np.testing.assert_allclose(observed[0].response, [-1.0, 1.0])
    np.testing.assert_allclose(observed[0].early_stopping_response.response, [2.0])
    assert model._pipeline._response_transform.mean_[0] == pytest.approx(2.0)


def test_empty_outputs_bypass_inverse_transform() -> None:
    class RejectEmpty:
        def inverse_transform(self, values):
            raise AssertionError("No valid predictions to invert")

    for values in (np.empty(0), np.array([np.nan, np.nan])):
        np.testing.assert_array_equal(apply_target_transform(RejectEmpty(), values, inverse=True), values)


def test_transformer_must_preserve_one_column_alignment() -> None:
    class WrongShape:
        def transform(self, values):
            return np.column_stack([values, values])

    with pytest.raises(ValueError, match="shape"):
        apply_target_transform(WrongShape(), np.array([1.0, np.nan, 2.0]))


def test_clone_cannot_return_the_prototype_itself() -> None:
    class SharedClone(_CenteringTransformer):
        def __sklearn_clone__(self):
            return self

    with pytest.raises(TypeError, match="independent instance"):
        fit_target_transform(SharedClone(), np.array([1.0, 2.0]))

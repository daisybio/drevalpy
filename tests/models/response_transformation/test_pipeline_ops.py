"""Fitting and applying a response transformation inside the model pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from drevalpy.errors import InvalidInputError
from drevalpy.models import DRPModel, ModelSpec, get_response_transformation
from drevalpy.models.response_transformation import GroupMeanCenterer, ResponseTransformer
from drevalpy.models.response_transformation._pipeline_ops import (
    apply_response_transformation,
    fit_response_transformation,
    group_labels,
)
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_model_dataset_gene_expression_fingerprints,
    synthetic_model_dataset_identity,
)


class _CenteringTransformer(BaseEstimator):
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
    fitted = fit_response_transformation(prototype, np.array([1.0, np.nan, 3.0]))

    assert fitted is not prototype
    transformed = apply_response_transformation(fitted, np.array([1.0, np.nan, 3.0]))
    assert np.isnan(transformed[1])
    np.testing.assert_allclose(
        apply_response_transformation(fitted, transformed, inverse=True), [1.0, np.nan, 3.0], equal_nan=True
    )
    values = np.array([1.0, np.nan, 2.0])
    np.testing.assert_array_equal(apply_response_transformation(None, values), values)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(TypeError, match="fit, transform, and inverse_transform"):
        fit_response_transformation(object(), np.array([1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_response_transformation(None, np.ones((2, 1)))


def test_sklearn_scalers_and_the_group_mean_centerer_are_response_transformers() -> None:
    assert isinstance(StandardScaler(), ResponseTransformer)
    assert isinstance(GroupMeanCenterer(), ResponseTransformer)
    assert not isinstance(object(), ResponseTransformer)


def test_a_transformer_without_inverse_transform_is_rejected_at_fit() -> None:
    class NoInverse:
        def fit(self, values):
            return self

        def transform(self, values):
            return values

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    assert not isinstance(NoInverse(), ResponseTransformer)
    with pytest.raises(TypeError, match="fit, transform, and inverse_transform"):
        fit_response_transformation(NoInverse(), np.array([1.0, 2.0]))


def test_custom_transform_is_independent_and_round_trips_with_retraining(tmp_path) -> None:
    prototype = _CenteringTransformer(offset=11.0)
    dataset = synthetic_model_dataset_identity()
    first = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=prototype)
    second = DRPModel(first.spec, response_transformation=prototype)
    scope = lso_split_masks().train
    first.train(dataset, scope)
    second.train(dataset, lso_split_masks().test)
    assert not hasattr(prototype, "mean_")
    assert first._pipeline._fitted_response_transformation is not second._pipeline._fitted_response_transformation
    np.testing.assert_allclose(first.predict(dataset, scope), 1.5)
    np.testing.assert_allclose(second.predict(dataset, scope), 3.5)
    first.save(tmp_path / "transformed")
    restored = DRPModel.load(tmp_path / "transformed")
    np.testing.assert_array_equal(restored.predict(dataset, scope), first.predict(dataset, scope))
    old_transform = restored._pipeline._fitted_response_transformation
    restored.train(dataset, lso_split_masks().test)
    assert restored._pipeline._fitted_response_transformation is not old_transform
    np.testing.assert_allclose(restored.predict(dataset, scope), 3.5)


def test_transform_fits_original_training_population_before_feature_filtering() -> None:
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    dataset.mdata.mod["gene_expression"].X[1, :] = np.nan
    model = DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler())
    model.train(dataset, SplitMask(np.ones((2, 2), dtype=bool)))
    assert model._pipeline._fitted_response_transformation.mean_[0] == pytest.approx(2.5)


def _recorded_fit(monkeypatch) -> list:
    from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor

    observed = []
    fit = NaiveMeanPredictor._fit
    monkeypatch.setattr(NaiveMeanPredictor, "_fit", lambda self, batch: (observed.append(batch), fit(self, batch))[1])
    return observed


def test_stopping_targets_are_transformed_using_training_statistics(monkeypatch) -> None:
    observed = _recorded_fit(monkeypatch)
    model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=StandardScaler())
    training = SplitMask.from_pairs(np.array([[0, 0], [1, 0]]), (2, 2))
    stopping = SplitMask.from_pairs(np.array([[1, 1]]), (2, 2))
    model.train(synthetic_model_dataset_identity(), training, early_stopping_scope=stopping)
    np.testing.assert_allclose(observed[0].response, [-1.0, 1.0])
    np.testing.assert_allclose(observed[0].early_stopping_response.response, [2.0])
    assert model._pipeline._fitted_response_transformation.mean_[0] == pytest.approx(2.0)


def test_transformer_contract_violations() -> None:
    class RejectEmpty:
        def inverse_transform(self, values):
            raise AssertionError("No valid predictions to invert")

    class WrongShape:
        def transform(self, values):
            return np.column_stack([values, values])

    class SharedClone(_CenteringTransformer):
        def __sklearn_clone__(self):
            return self

    for values in (np.empty(0), np.array([np.nan, np.nan])):
        np.testing.assert_array_equal(apply_response_transformation(RejectEmpty(), values, inverse=True), values)
    with pytest.raises(ValueError, match="shape"):
        apply_response_transformation(WrongShape(), np.array([1.0, np.nan, 2.0]))
    with pytest.raises(TypeError, match="independent instance"):
        fit_response_transformation(SharedClone(), np.array([1.0, 2.0]))


def _identity_dataset_with_tissues(tissues: list[str] | None):
    dataset = synthetic_model_dataset_identity()
    if tissues is not None:
        dataset.mdata.obs["tissue"] = tissues
    return dataset


class TestGroupLabels:
    @pytest.fixture
    def response(self) -> ResponseBatch:
        return ResponseBatch(np.array([1.0, 2.0]), np.array(["cl1", "cl2"]), np.array(["d2", "d1"]))

    def test_labels_per_transformer(self, response: ResponseBatch) -> None:
        identity = synthetic_model_dataset_identity()
        assert group_labels(StandardScaler(), identity, response) is None
        assert group_labels(None, identity, response) is None
        assert group_labels(GroupMeanCenterer(), identity, response).tolist() == [["d2"], ["d1"]]
        assert group_labels(GroupMeanCenterer(), _identity_dataset_with_tissues(None), response).shape == (2, 1)

        labels = group_labels(
            GroupMeanCenterer(("drug", "tissue")), _identity_dataset_with_tissues(["Lung", "Blood"]), response
        )
        assert labels.tolist() == [["d2", "Lung"], ["d1", "Blood"]]

    def test_invalid_datasets_and_levels_are_rejected(self, response: ResponseBatch) -> None:
        with pytest.raises(InvalidInputError, match="no 'tissue' sample column"):
            group_labels(GroupMeanCenterer(("drug", "tissue")), _identity_dataset_with_tissues(None), response)
        with pytest.raises(InvalidInputError, match="Unknown group level 'study'"):
            group_labels(GroupMeanCenterer(("study",)), synthetic_model_dataset_identity(), response)


class TestGroupsReachTheTransformer:
    def test_fit_and_apply_pass_the_groups_aligned_with_the_observed_values(self) -> None:
        values = np.array([1.0, np.nan, 3.0, 10.0])
        groups = np.array([["A"], ["A"], ["A"], ["B"]])

        fitted = fit_response_transformation(GroupMeanCenterer(), values, groups)
        transformed = apply_response_transformation(fitted, values, groups=groups)

        np.testing.assert_allclose(transformed, [-1.0, np.nan, 1.0, 0.0], equal_nan=True)
        np.testing.assert_allclose(
            apply_response_transformation(fitted, transformed, groups=groups, inverse=True), values, equal_nan=True
        )

    def test_scalers_ignore_groups_and_group_aware_transformers_fall_back_to_the_global_mean(self) -> None:
        groups = np.array([["A"], ["B"]])
        scaled = fit_response_transformation(StandardScaler(), np.array([1.0, 3.0]), groups)
        np.testing.assert_allclose(
            apply_response_transformation(scaled, np.array([1.0, 3.0]), groups=groups), [-1.0, 1.0]
        )

        values = np.array([1.0, 3.0, 8.0])
        fitted = fit_response_transformation(GroupMeanCenterer(), values)
        np.testing.assert_allclose(apply_response_transformation(fitted, values), [-3.0, -1.0, 4.0])


class TestGroupMeansInThePipeline:
    """On the 2x2 dataset ``[[1, 2], [3, 4]]`` (samples cl1 = Lung, cl2 = Blood; drugs d1, d2)."""

    def _predictions(self, name: str, train: SplitMask, predict_dataset=None) -> np.ndarray:
        dataset = _identity_dataset_with_tissues(["Lung", "Blood"])
        model = DRPModel(
            ModelSpec.from_name("NaivePredictor"), response_transformation=get_response_transformation(name)
        )
        model.train(dataset, train)
        return model.predict(predict_dataset or dataset, SplitMask(np.ones((2, 2), dtype=bool)))

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("none", [1.5] * 4),
            ("drug_mean", [1.0, 2.0, 1.0, 2.0]),
            ("drug_tissue_mean", [1.0, 2.0, 1.0, 2.0]),
        ],
        ids=["global-mean", "drug-mean", "unseen-tissue-falls-back-to-drug-mean"],
    )
    def test_trained_on_one_sample(self, name, expected) -> None:
        np.testing.assert_allclose(self._predictions(name, lso_split_masks().train), expected)

    def test_every_seen_combination_is_reproduced_exactly(self) -> None:
        everything = SplitMask(np.ones((2, 2), dtype=bool))

        np.testing.assert_allclose(self._predictions("drug_tissue_mean", everything), [1.0, 2.0, 3.0, 4.0])

    def test_an_unseen_drug_falls_back_to_the_global_mean(self) -> None:
        only_d1 = SplitMask.from_pairs(np.array([[0, 0], [1, 0]]), (2, 2))

        np.testing.assert_allclose(self._predictions("drug_mean", only_d1), 2.0)

    def test_the_tissue_is_looked_up_in_the_dataset_that_is_predicted(self) -> None:
        everything = SplitMask(np.ones((2, 2), dtype=bool))
        swapped = _identity_dataset_with_tissues(["Blood", "Lung"])

        predictions = self._predictions("drug_tissue_mean", everything, predict_dataset=swapped)

        np.testing.assert_allclose(predictions, [3.0, 4.0, 1.0, 2.0])

    def test_a_dataset_without_tissues_fails_with_a_clear_error(self) -> None:
        model = DRPModel(
            ModelSpec.from_name("NaivePredictor"), response_transformation=GroupMeanCenterer(("drug", "tissue"))
        )

        with pytest.raises(InvalidInputError, match="no 'tissue' sample column"):
            model.train(_identity_dataset_with_tissues(None), lso_split_masks().train)

    def test_the_fitted_transformer_travels_in_the_checkpoint(self, tmp_path) -> None:
        dataset = _identity_dataset_with_tissues(["Lung", "Blood"])
        scope = SplitMask(np.ones((2, 2), dtype=bool))
        model = DRPModel(
            ModelSpec.from_name("NaivePredictor"), response_transformation=GroupMeanCenterer(("drug", "tissue"))
        )
        model.train(dataset, scope)
        model.save(tmp_path / "grouped")

        restored = DRPModel.load(tmp_path / "grouped")

        assert isinstance(restored._pipeline._fitted_response_transformation, GroupMeanCenterer)
        np.testing.assert_array_equal(restored.predict(dataset, scope), model.predict(dataset, scope))
        np.testing.assert_allclose(restored.predict(dataset, scope), [1.0, 2.0, 3.0, 4.0])

    def test_the_early_stopping_targets_use_the_training_group_means(self, monkeypatch) -> None:
        seen = _recorded_fit(monkeypatch)
        training = SplitMask.from_pairs(np.array([[0, 0], [1, 0], [0, 1]]), (2, 2))
        stopping = SplitMask.from_pairs(np.array([[1, 1]]), (2, 2))
        model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=GroupMeanCenterer())

        model.train(synthetic_model_dataset_identity(), training, early_stopping_scope=stopping)

        np.testing.assert_allclose(seen[0].response, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(seen[0].early_stopping_response.response, [2.0])

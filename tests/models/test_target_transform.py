"""Tests for fitted target transformations used by the model pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from drevalpy.errors import InvalidInputError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models._target_transform import (
    apply_target_transform,
    fit_target_transform,
    group_labels,
    validate_fit_scopes,
)
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.utils.response_transform import GroupMeanCenterer, get_response_transformation
from tests.models.synthetic_fixtures import lso_split_masks, small_data_spec, synthetic_mudataset_identity


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
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    dataset = synthetic_mudataset_gene_expression_fingerprints()
    dataset.mdata.mod["gene_expression"].X[1, :] = np.nan
    model = DRPModel(small_data_spec("ElasticNet"), response_transformation=StandardScaler())
    model.train(dataset, SplitMask(np.ones((2, 2), dtype=bool)))
    assert model._pipeline._response_transform.mean_[0] == pytest.approx(2.5)


def test_stopping_targets_are_transformed_using_training_statistics(monkeypatch) -> None:
    from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor

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


def _identity_dataset_with_tissues(tissues: list[str] | None):
    """Return the 2x2 identity dataset (samples cl1, cl2) with *tissues* in its global sample metadata, or none.

    The fixture's own tissue column only exists as ``response:tissue``, which ``Dataset.get_tissue`` does not read.
    """
    dataset = synthetic_mudataset_identity()
    if tissues is not None:
        dataset.mdata.obs["tissue"] = tissues
    return dataset


class TestGroupLabels:
    """The label columns are built from the batch's drugs and the dataset's tissues."""

    @pytest.fixture
    def response(self) -> ResponseBatch:
        return ResponseBatch(np.array([1.0, 2.0]), np.array(["cl1", "cl2"]), np.array(["d2", "d1"]))

    def test_a_scaler_needs_no_groups(self, response: ResponseBatch) -> None:
        assert group_labels(StandardScaler(), synthetic_mudataset_identity(), response) is None
        assert group_labels(None, synthetic_mudataset_identity(), response) is None

    def test_drug_mean_labels_are_the_drug_ids(self, response: ResponseBatch) -> None:
        labels = group_labels(GroupMeanCenterer(), synthetic_mudataset_identity(), response)

        assert labels.tolist() == [["d2"], ["d1"]]

    def test_drug_tissue_mean_labels_add_the_tissue_of_each_sample(self, response: ResponseBatch) -> None:
        labels = group_labels(
            GroupMeanCenterer(("drug", "tissue")), _identity_dataset_with_tissues(["Lung", "Blood"]), response
        )

        assert labels.tolist() == [["d2", "Lung"], ["d1", "Blood"]]

    def test_a_dataset_without_tissues_is_rejected(self, response: ResponseBatch) -> None:
        dataset = _identity_dataset_with_tissues(None)

        with pytest.raises(InvalidInputError, match="no 'tissue' sample column"):
            group_labels(GroupMeanCenterer(("drug", "tissue")), dataset, response)

    def test_drug_mean_does_not_need_tissues(self, response: ResponseBatch) -> None:
        dataset = _identity_dataset_with_tissues(None)

        assert group_labels(GroupMeanCenterer(), dataset, response).shape == (2, 1)

    def test_an_unknown_level_is_rejected(self, response: ResponseBatch) -> None:
        with pytest.raises(InvalidInputError, match="Unknown group level 'study'"):
            group_labels(GroupMeanCenterer(("study",)), synthetic_mudataset_identity(), response)


class TestGroupsReachTheTransformer:
    def test_fit_and_apply_pass_the_groups_aligned_with_the_observed_values(self) -> None:
        values = np.array([1.0, np.nan, 3.0, 10.0])
        groups = np.array([["A"], ["A"], ["A"], ["B"]])

        fitted = fit_target_transform(GroupMeanCenterer(), values, groups)
        transformed = apply_target_transform(fitted, values, groups=groups)

        np.testing.assert_allclose(transformed, [-1.0, np.nan, 1.0, 0.0], equal_nan=True)
        np.testing.assert_allclose(
            apply_target_transform(fitted, transformed, groups=groups, inverse=True), values, equal_nan=True
        )

    def test_a_scaler_is_never_given_groups(self) -> None:
        values = np.array([1.0, 3.0])
        groups = np.array([["A"], ["B"]])

        fitted = fit_target_transform(StandardScaler(), values, groups)

        np.testing.assert_allclose(apply_target_transform(fitted, values, groups=groups), [-1.0, 1.0])

    def test_without_groups_a_group_aware_transformer_centres_on_the_global_mean(self) -> None:
        values = np.array([1.0, 3.0, 8.0])

        fitted = fit_target_transform(GroupMeanCenterer(), values)

        np.testing.assert_allclose(apply_target_transform(fitted, values), [-3.0, -1.0, 4.0])


class TestGroupMeansInThePipeline:
    """On the 2x2 dataset ``[[1, 2], [3, 4]]`` (samples cl1 = Lung, cl2 = Blood; drugs d1, d2)."""

    def _predictions(self, name: str, train: SplitMask, dataset=None, predict_dataset=None) -> np.ndarray:
        dataset = dataset or _identity_dataset_with_tissues(["Lung", "Blood"])
        model = DRPModel(
            ModelSpec.from_name("NaivePredictor"), response_transformation=get_response_transformation(name)
        )
        model.train(dataset, train)
        return model.predict(predict_dataset or dataset, SplitMask(np.ones((2, 2), dtype=bool)))

    def test_drug_mean_makes_the_naive_mean_predict_the_drug_mean(self) -> None:
        train = lso_split_masks().train

        np.testing.assert_allclose(self._predictions("drug_mean", train), [1.0, 2.0, 1.0, 2.0])

    def test_the_global_mean_is_still_what_a_plain_run_predicts(self) -> None:
        np.testing.assert_allclose(self._predictions("none", lso_split_masks().train), 1.5)

    def test_an_unseen_tissue_falls_back_to_the_drug_mean(self) -> None:
        # Trained on cl1 (Lung) only, so Blood was never seen and cl2 is centred on its drug's mean.
        predictions = self._predictions("drug_tissue_mean", lso_split_masks().train)

        np.testing.assert_allclose(predictions, [1.0, 2.0, 1.0, 2.0])

    def test_every_seen_combination_is_reproduced_exactly(self) -> None:
        everything = SplitMask(np.ones((2, 2), dtype=bool))

        np.testing.assert_allclose(self._predictions("drug_tissue_mean", everything), [1.0, 2.0, 3.0, 4.0])

    def test_an_unseen_drug_falls_back_to_the_global_mean(self) -> None:
        only_d1 = SplitMask.from_pairs(np.array([[0, 0], [1, 0]]), (2, 2))

        predictions = self._predictions("drug_mean", only_d1)

        # d1 has mean 2; the naive mean of the residuals is 0, and d2 is centred on the global mean 2.
        np.testing.assert_allclose(predictions, 2.0)

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

        assert isinstance(restored._pipeline._response_transform, GroupMeanCenterer)
        np.testing.assert_array_equal(restored.predict(dataset, scope), model.predict(dataset, scope))
        np.testing.assert_allclose(restored.predict(dataset, scope), [1.0, 2.0, 3.0, 4.0])

    def test_the_early_stopping_targets_use_the_training_group_means(self, monkeypatch) -> None:
        from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor

        seen = []
        fit = NaiveMeanPredictor._fit
        monkeypatch.setattr(NaiveMeanPredictor, "_fit", lambda self, batch: (seen.append(batch), fit(self, batch))[1])
        dataset = synthetic_mudataset_identity()
        training = SplitMask.from_pairs(np.array([[0, 0], [1, 0], [0, 1]]), (2, 2))
        stopping = SplitMask.from_pairs(np.array([[1, 1]]), (2, 2))
        model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=GroupMeanCenterer())

        model.train(dataset, training, early_stopping_scope=stopping)

        np.testing.assert_allclose(seen[0].response, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(seen[0].early_stopping_response.response, [2.0])

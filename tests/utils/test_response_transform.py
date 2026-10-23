"""Tests for the sklearn response-transformation lookup."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.utils.response_transform import GroupMeanCenterer, get_response_transformation


@pytest.mark.parametrize(
    "option",
    [
        pytest.param("None", id="literal-none-string"),
        pytest.param("none", id="lowercase-none-string"),
        pytest.param("NONE", id="uppercase-none-string"),
        pytest.param(None, id="none-object"),
    ],
)
def test_no_transformation_requested_returns_none(option: str | None) -> None:
    assert get_response_transformation(option) is None


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        pytest.param("standard", StandardScaler, id="standard"),
        pytest.param("minmax", MinMaxScaler, id="minmax"),
        pytest.param("robust", RobustScaler, id="robust"),
    ],
)
def test_known_options_return_the_matching_scaler(option: str, expected: type) -> None:
    assert isinstance(get_response_transformation(option), expected)


def test_each_call_returns_a_fresh_unfitted_transformer() -> None:
    first = get_response_transformation("standard")
    second = get_response_transformation("standard")

    assert first is not second
    assert not hasattr(first, "mean_")


def test_returned_transformer_is_usable() -> None:
    transformer = get_response_transformation("minmax")

    scaled = transformer.fit_transform(np.array([[0.0], [5.0], [10.0]]))

    np.testing.assert_allclose(scaled.ravel(), [0.0, 0.5, 1.0])


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(UnknownNameError, match="Unknown response transformation quantile"):
        get_response_transformation("quantile")


def test_rejection_message_lists_the_valid_options() -> None:
    with pytest.raises(ValueError, match="'None', 'standard', 'minmax', 'robust', 'drug_mean', 'drug_tissue_mean'"):
        get_response_transformation("Standard")


def test_drug_mean_option_centres_on_the_drug() -> None:
    transformer = get_response_transformation("drug_mean")

    assert isinstance(transformer, GroupMeanCenterer)
    assert transformer.levels == ("drug",)


def test_drug_tissue_mean_option_centres_on_drug_and_tissue() -> None:
    transformer = get_response_transformation("drug_tissue_mean")

    assert isinstance(transformer, GroupMeanCenterer)
    assert transformer.levels == ("drug", "tissue")


_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 5.0])
_DRUGS = np.array(["A", "A", "B", "B", "B", "C"])
#: means of _RESPONSE per group in _DRUGS
_GROUP_MEANS = {"A": 2.0, "B": 20.0, "C": 5.0}

_TT_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 40.0, 7.0, 9.0])
_TT_DRUGS = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
_TT_TISSUES = np.array(["lung", "lung", "skin", "skin", "lung", "lung", "skin", "skin"])
_TT_GROUPS = np.stack([_TT_DRUGS, _TT_TISSUES], axis=1)
#: means of _TT_RESPONSE per (drug, tissue) combination
_TT_MEANS = {("A", "lung"): 2.0, ("A", "skin"): 15.0, ("B", "lung"): 35.0, ("B", "skin"): 8.0}
#: means of _TT_RESPONSE per drug, i.e. the next level of the fallback
_TT_DRUG_MEANS = {"A": 8.5, "B": 21.5}


def _tissue_centerer() -> GroupMeanCenterer:
    return GroupMeanCenterer(("drug", "tissue")).fit(_TT_RESPONSE.reshape(-1, 1), groups=_TT_GROUPS)


class TestGroupMeanCentererOneLevel:
    def test_it_asks_the_pipeline_for_groups(self) -> None:
        assert GroupMeanCenterer.requires_groups is True
        assert getattr(StandardScaler, "requires_groups", False) is False

    def test_fit_estimates_group_means(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)

        assert np.isclose(centerer.global_mean_, _RESPONSE.mean())
        assert len(centerer.level_keys_) == 1
        assert list(centerer.level_keys_[0]) == ["A", "B", "C"]
        assert np.allclose(centerer.level_means_[0], [_GROUP_MEANS[key] for key in ["A", "B", "C"]])

    def test_transform_subtracts_the_group_mean(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
        residuals = centerer.transform(_RESPONSE.reshape(-1, 1), groups=_DRUGS).squeeze()

        assert np.allclose(residuals, _RESPONSE - np.array([_GROUP_MEANS[drug] for drug in _DRUGS]))
        for drug in np.unique(_DRUGS):
            assert np.isclose(residuals[_DRUGS == drug].mean(), 0.0)

    def test_inverse_transform_round_trips(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
        residuals = centerer.transform(_RESPONSE.reshape(-1, 1), groups=_DRUGS)

        assert np.allclose(centerer.inverse_transform(residuals, groups=_DRUGS).squeeze(), _RESPONSE)

    def test_the_shape_of_the_input_is_kept(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
        column = _RESPONSE.reshape(-1, 1)

        assert centerer.transform(column, groups=_DRUGS).shape == column.shape
        assert centerer.transform(_RESPONSE, groups=_DRUGS).shape == _RESPONSE.shape

    def test_an_unseen_drug_falls_back_to_the_global_mean(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
        # "0" sorts before and "Z" after every fitted key, which is where searchsorted needs clipping.
        unseen = np.array(["Z", "0", "A"])
        values = np.array([1.0, 1.0, 1.0])

        residuals = centerer.transform(values, groups=unseen)

        assert np.allclose(residuals, values - np.array([centerer.global_mean_, centerer.global_mean_, 2.0]))
        assert np.allclose(centerer.inverse_transform(residuals, groups=unseen), values)

    def test_without_groups_it_is_plain_mean_centering(self) -> None:
        centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1))

        assert centerer.level_keys_ == []
        assert np.allclose(centerer.transform(_RESPONSE), _RESPONSE - _RESPONSE.mean())

    def test_fitting_on_nothing_gives_a_zero_offset(self) -> None:
        centerer = GroupMeanCenterer().fit(np.array([]).reshape(-1, 1), groups=np.array([]))

        assert centerer.global_mean_ == 0.0
        assert centerer.transform(np.array([])).size == 0

    def test_an_unfitted_transformer_refuses_to_transform(self) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            GroupMeanCenterer().transform(_RESPONSE, groups=_DRUGS)


class TestGroupMeanCentererTwoLevels:
    def test_every_combination_is_centred_on_zero(self) -> None:
        centerer = _tissue_centerer()
        residuals = centerer.transform(_TT_RESPONSE, groups=_TT_GROUPS)

        assert np.allclose(
            residuals,
            _TT_RESPONSE
            - np.array([_TT_MEANS[(drug, tissue)] for drug, tissue in zip(_TT_DRUGS, _TT_TISSUES, strict=True)]),
        )
        assert np.allclose(centerer.inverse_transform(residuals, groups=_TT_GROUPS), _TT_RESPONSE)

    def test_fit_estimates_both_nesting_levels(self) -> None:
        centerer = _tissue_centerer()

        assert len(centerer.level_keys_) == 2
        assert list(centerer.level_keys_[1]) == ["A", "B"]
        assert np.allclose(centerer.level_means_[1], [_TT_DRUG_MEANS["A"], _TT_DRUG_MEANS["B"]])
        assert np.isclose(centerer.global_mean_, _TT_RESPONSE.mean())

    def test_an_unseen_tissue_falls_back_to_the_drug_mean_and_an_unseen_drug_to_the_global_mean(self) -> None:
        centerer = _tissue_centerer()
        # "liver" is unknown for both drugs, "Z" is an unknown drug, "0" and "z" bracket the fitted keys.
        groups = np.array([["A", "liver"], ["B", "liver"], ["Z", "lung"], ["0", "0"], ["z", "z"]])
        values = np.ones(len(groups))

        offsets = values - centerer.transform(values, groups=groups)

        assert np.allclose(offsets, [8.5, 21.5] + [centerer.global_mean_] * 3)
        assert np.allclose(centerer.inverse_transform(centerer.transform(values, groups=groups), groups=groups), values)

    def test_a_seen_drug_in_an_unseen_tissue_is_not_centred_on_the_global_mean(self) -> None:
        centerer = _tissue_centerer()

        offset = 1.0 - centerer.transform(np.ones(1), groups=np.array([["A", "liver"]]))

        assert offset[0] != pytest.approx(centerer.global_mean_)

    def test_fewer_group_columns_than_at_fit_time_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="group level"):
            _tissue_centerer().transform(_TT_RESPONSE, groups=_TT_DRUGS)

    def test_ambiguous_concatenations_stay_distinct(self) -> None:
        responses = np.array([2.0, 8.0])
        groups = np.array([["A", "B_C"], ["A_B", "C"]])
        centerer = GroupMeanCenterer(("drug", "tissue")).fit(responses.reshape(-1, 1), groups=groups)

        assert np.allclose(centerer.transform(responses, groups=groups), [0.0, 0.0])

    def test_the_label_columns_have_to_match_the_levels(self) -> None:
        with pytest.raises(InvalidInputError, match="group column"):
            GroupMeanCenterer(("drug", "tissue")).fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)

    def test_an_unknown_level_is_rejected_at_fit(self) -> None:
        with pytest.raises(InvalidInputError, match="Unknown group level 'study'"):
            GroupMeanCenterer(("study",)).fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)


class TestGroupMeanCentererAsAnEstimator:
    def test_clone_gives_an_unfitted_copy_with_the_same_levels(self) -> None:
        fitted = _tissue_centerer()

        copy = clone(fitted)

        assert copy is not fitted
        assert copy.levels == ("drug", "tissue")
        assert not hasattr(copy, "global_mean_")

    def test_set_params_changes_the_levels(self) -> None:
        assert GroupMeanCenterer().set_params(levels=("drug", "tissue")).get_params() == {"levels": ("drug", "tissue")}

    def test_set_params_rejects_an_unknown_name(self) -> None:
        with pytest.raises(InvalidInputError, match="Invalid parameter 'depth'"):
            GroupMeanCenterer().set_params(depth=2)

    def test_a_fitted_transformer_survives_pickling(self) -> None:
        restored = pickle.loads(pickle.dumps(_tissue_centerer()))  # noqa: S301 -- the object was pickled one line earlier

        assert np.allclose(
            restored.transform(_TT_RESPONSE, groups=_TT_GROUPS),
            _tissue_centerer().transform(_TT_RESPONSE, groups=_TT_GROUPS),
        )

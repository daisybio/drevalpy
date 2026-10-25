"""The group-mean centerer."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from drevalpy.errors import InvalidInputError
from drevalpy.models.response_transformation import GroupMeanCenterer

_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 5.0])
_DRUGS = np.array(["A", "A", "B", "B", "B", "C"])
_GROUP_MEANS = {"A": 2.0, "B": 20.0, "C": 5.0}

_TT_RESPONSE = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 40.0, 7.0, 9.0])
_TT_DRUGS = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
_TT_TISSUES = np.array(["lung", "lung", "skin", "skin", "lung", "lung", "skin", "skin"])
_TT_GROUPS = np.stack([_TT_DRUGS, _TT_TISSUES], axis=1)
_TT_MEANS = {("A", "lung"): 2.0, ("A", "skin"): 15.0, ("B", "lung"): 35.0, ("B", "skin"): 8.0}
_TT_DRUG_MEANS = {"A": 8.5, "B": 21.5}


def _tissue_centerer() -> GroupMeanCenterer:
    return GroupMeanCenterer(("drug", "tissue")).fit(_TT_RESPONSE.reshape(-1, 1), groups=_TT_GROUPS)


def test_one_level_fit_transform_and_inverse() -> None:
    assert GroupMeanCenterer.requires_groups is True
    assert getattr(StandardScaler, "requires_groups", False) is False
    column = _RESPONSE.reshape(-1, 1)
    centerer = GroupMeanCenterer().fit(column, groups=_DRUGS)

    assert np.isclose(centerer.global_mean_, _RESPONSE.mean())
    assert list(centerer.level_keys_[0]) == ["A", "B", "C"]
    assert np.allclose(centerer.level_means_[0], [_GROUP_MEANS[key] for key in "ABC"])

    residuals = centerer.transform(column, groups=_DRUGS)
    assert residuals.shape == column.shape
    assert centerer.transform(_RESPONSE, groups=_DRUGS).shape == _RESPONSE.shape
    assert np.allclose(residuals.squeeze(), _RESPONSE - np.array([_GROUP_MEANS[drug] for drug in _DRUGS]))
    assert np.allclose(centerer.inverse_transform(residuals, groups=_DRUGS).squeeze(), _RESPONSE)


def test_an_unseen_drug_falls_back_to_the_global_mean() -> None:
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1), groups=_DRUGS)
    unseen = np.array(["Z", "0", "A"])  # "0" sorts before and "Z" after every fitted key (searchsorted clipping)
    values = np.ones(3)
    residuals = centerer.transform(values, groups=unseen)
    assert np.allclose(residuals, values - np.array([centerer.global_mean_, centerer.global_mean_, 2.0]))
    assert np.allclose(centerer.inverse_transform(residuals, groups=unseen), values)


def test_without_groups_or_data_edge_cases() -> None:
    centerer = GroupMeanCenterer().fit(_RESPONSE.reshape(-1, 1))
    assert centerer.level_keys_ == []
    assert np.allclose(centerer.transform(_RESPONSE), _RESPONSE - _RESPONSE.mean())

    empty = GroupMeanCenterer().fit(np.array([]).reshape(-1, 1), groups=np.array([]))
    assert empty.global_mean_ == 0.0
    assert empty.transform(np.array([])).size == 0

    with pytest.raises(ValueError, match="not fitted"):
        GroupMeanCenterer().transform(_RESPONSE, groups=_DRUGS)


def test_two_levels_centre_every_combination_and_fit_both_levels() -> None:
    centerer = _tissue_centerer()
    residuals = centerer.transform(_TT_RESPONSE, groups=_TT_GROUPS)
    expected = np.array([_TT_MEANS[key] for key in zip(_TT_DRUGS, _TT_TISSUES, strict=True)])
    assert np.allclose(residuals, _TT_RESPONSE - expected)
    assert np.allclose(centerer.inverse_transform(residuals, groups=_TT_GROUPS), _TT_RESPONSE)
    assert list(centerer.level_keys_[1]) == ["A", "B"]
    assert np.allclose(centerer.level_means_[1], [_TT_DRUG_MEANS["A"], _TT_DRUG_MEANS["B"]])
    assert np.isclose(centerer.global_mean_, _TT_RESPONSE.mean())


def test_unseen_tissue_falls_back_to_the_drug_mean_and_unseen_drug_to_the_global_mean() -> None:
    centerer = _tissue_centerer()
    groups = np.array([["A", "liver"], ["B", "liver"], ["Z", "lung"], ["0", "0"], ["z", "z"]])
    values = np.ones(len(groups))
    offsets = values - centerer.transform(values, groups=groups)
    assert np.allclose(offsets, [8.5, 21.5] + [centerer.global_mean_] * 3)
    assert not np.isclose(offsets[0], centerer.global_mean_)
    assert np.allclose(centerer.inverse_transform(centerer.transform(values, groups=groups), groups=groups), values)


def test_ambiguous_concatenations_stay_distinct() -> None:
    responses = np.array([2.0, 8.0])
    groups = np.array([["A", "B_C"], ["A_B", "C"]])
    centerer = GroupMeanCenterer(("drug", "tissue")).fit(responses.reshape(-1, 1), groups=groups)
    assert np.allclose(centerer.transform(responses, groups=groups), [0.0, 0.0])


@pytest.mark.parametrize(
    ("levels", "groups", "match"),
    [
        (("drug", "tissue"), _DRUGS, "group column"),
        (("study",), _DRUGS, "Unknown group level 'study'"),
    ],
)
def test_invalid_group_setup_is_rejected_at_fit(levels, groups, match) -> None:
    with pytest.raises(InvalidInputError, match=match):
        GroupMeanCenterer(levels).fit(_RESPONSE.reshape(-1, 1), groups=groups)


def test_fewer_group_columns_than_at_fit_time_are_rejected() -> None:
    with pytest.raises(ValueError, match="group level"):
        _tissue_centerer().transform(_TT_RESPONSE, groups=_TT_DRUGS)


def test_fit_transform_passes_the_groups_to_fit_and_transform() -> None:
    column = _TT_RESPONSE.reshape(-1, 1)
    combined = GroupMeanCenterer(("drug", "tissue")).fit_transform(column, groups=_TT_GROUPS)
    separate = _tissue_centerer().transform(column, groups=_TT_GROUPS)
    np.testing.assert_array_equal(combined, separate)
    expected = _TT_RESPONSE - np.array([_TT_MEANS[key] for key in zip(_TT_DRUGS, _TT_TISSUES, strict=True)])
    np.testing.assert_allclose(combined.squeeze(), expected)


def test_clone_is_an_independent_unfitted_copy() -> None:
    fitted = _tissue_centerer()
    copy = clone(fitted)
    assert copy is not fitted
    assert copy.levels == ("drug", "tissue")
    assert not hasattr(copy, "global_mean_")
    copy.fit(_RESPONSE.reshape(-1, 1), groups=np.stack([_DRUGS, _DRUGS], axis=1))
    assert fitted.global_mean_ == pytest.approx(_TT_RESPONSE.mean())


def test_parameters_follow_the_sklearn_estimator_protocol() -> None:
    assert GroupMeanCenterer().set_params(levels=("drug", "tissue")).get_params() == {"levels": ("drug", "tissue")}
    with pytest.raises(ValueError, match="Invalid parameter 'depth'"):
        GroupMeanCenterer().set_params(depth=2)


def test_sklearn_tags_describe_a_transformer() -> None:
    tags = GroupMeanCenterer().__sklearn_tags__()
    assert tags.transformer_tags is not None


def test_pickling_keeps_the_fitted_state() -> None:
    fitted = _tissue_centerer()
    restored = pickle.loads(pickle.dumps(fitted))  # noqa: S301 -- pickled one line earlier
    assert np.allclose(
        restored.transform(_TT_RESPONSE, groups=_TT_GROUPS), fitted.transform(_TT_RESPONSE, groups=_TT_GROUPS)
    )

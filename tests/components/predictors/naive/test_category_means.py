"""Tests for the category-code helpers shared by the naive mean predictors."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drevalpy.components.predictors.naive._category_means import (
    additive_effects,
    block_pair_codes,
    effects_of,
    joint_codes,
    pair_codes,
    require_pair_codes,
)
from drevalpy.types.data.feature_source import Side
from tests.components.predictors.naive._helpers import naive_batch, one_hot

AB = one_hot(["a", "b"], ["a", "b"])


@pytest.mark.parametrize(
    ("matrix", "idx", "expected", "n_columns"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], [1, 0, 1], [1, 0, 1], 2),
        ([[0.0, 0.0], [0.0, 1.0]], [0, 1, 1], [-1, 1, 1], 2),
        ([1.0, 0.0], [0, 1, 0], [0, -1, 0], 1),
        (np.empty((2, 0)), [0, 1], [-1, -1], 0),
    ],
    ids=["one-hot", "all-zero-row-is-no-category", "1d-vector", "no-columns"],
)
def test_pair_codes_give_each_pair_the_column_of_its_entity(matrix, idx, expected, n_columns) -> None:
    codes, n = pair_codes(np.array(matrix), np.array(idx))

    assert codes.tolist() == expected
    assert n == n_columns


def test_pair_codes_requires_a_pair_index() -> None:
    with pytest.raises(ValueError, match="pair index is required"):
        pair_codes(np.eye(2), None)


@pytest.mark.parametrize("matrix", [[[1.0, 1.0]], [[0.5, 0.0]], [[2.0, 0.0]]], ids=["two-ones", "fraction", "two"])
def test_pair_codes_reject_rows_that_are_not_one_hot(matrix) -> None:
    with pytest.raises(ValueError, match="need one-hot rows"):
        pair_codes(np.array(matrix), np.array([0]))


def test_require_pair_codes_aligns_each_side() -> None:
    batch = naive_batch(
        sample_features=AB,
        drug_features=AB,
        sample_pair_idx=np.array([0, 1, 0], dtype=np.int64),
        drug_pair_idx=np.array([1, 1, 1], dtype=np.int64),
    )

    assert require_pair_codes(batch, side=Side.SAMPLE)[0].tolist() == [0, 1, 0]
    assert require_pair_codes(batch, side=Side.DRUG)[0].tolist() == [1, 1, 1]


def test_require_pair_codes_rejects_a_batch_without_drug_features() -> None:
    batch = naive_batch(sample_features=AB, sample_pair_idx=np.array([0, 1], dtype=np.int64))

    with pytest.raises(ValueError, match="drug features are required"):
        require_pair_codes(replace(batch, drug=None), side=Side.DRUG)


def test_block_pair_codes_aligns_named_block_and_rejects_missing() -> None:
    batch = naive_batch(
        sample_features=AB,
        sample_pair_idx=np.array([1, 0], dtype=np.int64),
        sample_blocks={"oneHot": AB},
    )

    codes, n = block_pair_codes(batch, "oneHot")
    assert (codes.tolist(), n) == ([1, 0], 2)
    with pytest.raises(ValueError, match="Required sample block 'tissue' is missing"):
        block_pair_codes(batch, "tissue")


def test_joint_codes_number_combinations_row_major_and_drop_pairs_missing_either_category() -> None:
    codes, n = joint_codes(np.array([0, 1, -1, 1]), 2, np.array([2, 0, 1, -1]), 3)

    assert codes.tolist() == [2, 3, -1, -1]
    assert n == 6


@pytest.mark.parametrize(
    ("labels", "response", "expected"),
    [(["a", "b"], [2.0, 6.0], [-2.0, 2.0]), (["a", "a"], [2.0, 6.0], [0.0, 0.0])],
)
def test_additive_effects_are_means_minus_baseline_and_zero_when_unobserved(labels, response, expected) -> None:
    codes, n = pair_codes(AB, np.array([{"a": 0, "b": 1}[label] for label in labels]))

    np.testing.assert_allclose(additive_effects(codes, n, np.array(response), baseline=4.0), expected)


def test_additive_effects_skip_pairs_without_a_category_and_ignore_the_response_of_unknown_entities() -> None:
    codes = np.array([0, -1, 0, 1, -1])

    result = additive_effects(codes, 3, np.array([1.0, 1000.0, 3.0, 9.0, -1000.0]), baseline=1.0)

    np.testing.assert_allclose(result, [1.0, 8.0, 0.0])


def test_additive_effects_reject_a_response_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="must match response"):
        additive_effects(np.array([0, 1, 0]), 2, np.array([1.0, 2.0]), baseline=0.0)


def test_zero_width_codes_yield_empty_effects_and_baseline_predictions() -> None:
    codes = np.full(3, -1)

    assert additive_effects(codes, 0, np.array([1.0, 2.0, 3.0]), baseline=1.5).shape == (0,)
    np.testing.assert_allclose(effects_of(codes, np.empty(0)), [0.0] * 3)


def test_effects_of_selects_each_category_effect_and_nothing_for_no_category() -> None:
    codes = np.array([0, 1, 0, -1])

    np.testing.assert_allclose(effects_of(codes, np.array([-1.0, 2.0])), [-1.0, 2.0, -1.0, 0.0])


def test_effects_of_rejects_codes_beyond_the_effects() -> None:
    with pytest.raises(ValueError, match="must match the number of categories"):
        effects_of(np.array([0, 2]), np.array([1.0, 2.0]))

"""Tests for the matrix helpers shared by the naive mean predictors."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    block_pair_matrix,
    category_means,
    pair_align,
    predict_with_effects,
    require_pair_matrix,
    state_float_matrix,
    state_float_vector,
)
from tests.components.predictors.naive._helpers import naive_batch, one_hot

AB = one_hot(["a", "b"], ["a", "b"])


@pytest.mark.parametrize(
    ("matrix", "idx", "expected"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], [1, 0, 1], [[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]),
        ([5.0, 6.0], [0, 1, 1], [[5.0], [6.0], [6.0]]),
    ],
)
def test_pair_align_expands_entity_rows_and_promotes_vectors(matrix, idx, expected) -> None:
    np.testing.assert_allclose(pair_align(np.array(matrix), np.array(idx)), expected)


def test_pair_align_requires_a_pair_index() -> None:
    with pytest.raises(ValueError, match="pair index is required"):
        pair_align(np.ones((2, 2)), None)


def test_require_pair_matrix_aligns_each_side_and_rejects_unknown_sides() -> None:
    batch = naive_batch(
        sample_features=AB,
        drug_features=AB,
        sample_pair_idx=np.array([0, 1, 0], dtype=np.int64),
        drug_pair_idx=np.array([1, 1, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(require_pair_matrix(batch, side="sample"), [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(require_pair_matrix(batch, side="drug"), [[0, 1]] * 3)
    with pytest.raises(ValueError, match="Unknown feature side"):
        require_pair_matrix(batch, side="tissue")


def test_block_pair_matrix_aligns_named_block_and_rejects_missing() -> None:
    batch = naive_batch(
        sample_features=AB,
        sample_pair_idx=np.array([1, 0], dtype=np.int64),
        sample_blocks={"identity": AB},
    )

    np.testing.assert_allclose(block_pair_matrix(batch, "identity"), [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="Required sample block 'tissue' is missing"):
        block_pair_matrix(batch, "tissue")


@pytest.mark.parametrize(
    ("labels", "response", "expected"),
    [(["a", "b", "a"], [1.0, 10.0, 3.0], [2.0, 10.0]), (["a", "a"], [4.0, 6.0], [5.0, 0.0])],
)
def test_category_means_average_per_column_and_zero_unobserved(labels, response, expected) -> None:
    result = category_means(one_hot(labels, ["a", "b"]), np.array(response))

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("design", "response", "match"),
    [(np.array([1.0, 2.0]), [1.0, 2.0], "must be 2-dimensional"), (np.ones((3, 2)), [1.0, 2.0], "must match response")],
)
def test_category_means_rejects_malformed_inputs(design, response, match) -> None:
    with pytest.raises(ValueError, match=match):
        category_means(design, np.array(response))


def test_zero_width_designs_yield_empty_effects_and_baseline_predictions() -> None:
    empty = np.empty((3, 0))

    assert category_means(empty, np.array([1.0, 2.0, 3.0])).shape == (0,)
    assert additive_effects(empty, np.array([1.0, 2.0, 3.0]), baseline=1.5).shape == (0,)
    np.testing.assert_allclose(predict_with_effects(empty, np.empty(0), baseline=2.5), [2.5] * 3)


@pytest.mark.parametrize(
    ("labels", "response", "expected"),
    [(["a", "b"], [2.0, 6.0], [-2.0, 2.0]), (["a", "a"], [2.0, 6.0], [0.0, 0.0])],
)
def test_additive_effects_are_means_minus_baseline_and_zero_when_unobserved(labels, response, expected) -> None:
    result = additive_effects(one_hot(labels, ["a", "b"]), np.array(response), baseline=4.0)

    np.testing.assert_allclose(result, expected)


def test_predict_with_effects_adds_selected_effects_to_the_baseline() -> None:
    design = one_hot(["a", "b", "a"], ["a", "b"])

    np.testing.assert_allclose(predict_with_effects(design, np.array([-1.0, 2.0]), baseline=5.0), [4.0, 7.0, 4.0])


@pytest.mark.parametrize(
    ("design", "effects", "match"),
    [(np.array([1.0, 0.0]), [1.0, 2.0], "must be 2-dimensional"), (np.ones((2, 3)), [1.0, 2.0], "must match design")],
)
def test_predict_with_effects_rejects_malformed_inputs(design, effects, match) -> None:
    with pytest.raises(ValueError, match=match):
        predict_with_effects(design, np.array(effects), baseline=0.0)


def test_state_float_vector_flattens_to_float64_and_returns_none_when_absent() -> None:
    result = state_float_vector({"effects": [[1.0], [2.0]]}, "effects")

    assert result is not None
    np.testing.assert_allclose(result, [1.0, 2.0])
    assert result.dtype == np.float64
    assert state_float_vector({}, "effects") is None


@pytest.mark.parametrize(("payload", "shape"), [([1.0, 2.0, 3.0], (3, 1)), ([[1.0, 2.0], [3.0, 4.0]], (2, 2))])
def test_state_float_matrix_promotes_flat_lists_and_keeps_2d(payload, shape) -> None:
    result = state_float_matrix({"table": payload}, "table")

    assert result is not None
    assert result.shape == shape
    assert result.dtype == np.float64
    assert state_float_matrix({"table": None}, "table") is None

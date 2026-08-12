"""Tests for the matrix helpers shared by the naive mean predictors.

Mirrors the private module
``drevalpy.components.predictors.naive._matrix_means`` with the leading
underscore stripped (``AGENTS.md`` rule 4: the ``naive`` package exposes no
public module for these helpers). All eight functions are exercised here
directly rather than only indirectly through ``effects.py`` / ``mean.py``.
"""

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


def test_pair_align_expands_entity_rows_to_pair_rows() -> None:
    entity_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = pair_align(entity_matrix, np.array([1, 0, 1]))

    np.testing.assert_allclose(result, [[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])


def test_pair_align_promotes_a_one_dimensional_matrix_to_a_column() -> None:
    result = pair_align(np.array([5.0, 6.0]), np.array([0, 1, 1]))

    assert result.shape == (3, 1)
    np.testing.assert_allclose(result, [[5.0], [6.0], [6.0]])


def test_pair_align_requires_a_pair_index() -> None:
    with pytest.raises(ValueError, match="pair index is required"):
        pair_align(np.ones((2, 2)), None)


def test_require_pair_matrix_aligns_the_cell_line_side() -> None:
    batch = naive_batch(
        cell_line_features=one_hot(["cl1", "cl2"], ["cl1", "cl2"]),
        cell_line_pair_idx=np.array([0, 1, 0], dtype=np.int64),
        n_pairs=3,
    )

    result = require_pair_matrix(batch, side="cell_line")

    np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])


def test_require_pair_matrix_aligns_the_drug_side() -> None:
    batch = naive_batch(
        drug_features=one_hot(["d1", "d2"], ["d1", "d2"]),
        drug_pair_idx=np.array([1, 1], dtype=np.int64),
        n_pairs=2,
    )

    result = require_pair_matrix(batch, side="drug")

    np.testing.assert_allclose(result, [[0.0, 1.0], [0.0, 1.0]])


def test_require_pair_matrix_rejects_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="Unknown feature side"):
        require_pair_matrix(naive_batch(n_pairs=1), side="tissue")


def test_block_pair_matrix_aligns_a_named_cell_line_block() -> None:
    identity = one_hot(["cl1", "cl2"], ["cl1", "cl2"])
    batch = naive_batch(
        cell_line_features=identity,
        cell_line_pair_idx=np.array([1, 0], dtype=np.int64),
        cell_line_blocks={"identity": identity},
        n_pairs=2,
    )

    result = block_pair_matrix(batch, "identity")

    np.testing.assert_allclose(result, [[0.0, 1.0], [1.0, 0.0]])


def test_block_pair_matrix_rejects_a_missing_block() -> None:
    batch = naive_batch(n_pairs=1)

    with pytest.raises(ValueError, match="Required cell-line block 'tissue' is missing"):
        block_pair_matrix(batch, "tissue")


def test_category_means_averages_the_response_per_one_hot_column() -> None:
    design = one_hot(["a", "b", "a"], ["a", "b"])

    result = category_means(design, np.array([1.0, 10.0, 3.0]))

    np.testing.assert_allclose(result, [2.0, 10.0])


def test_category_means_returns_zero_for_unobserved_columns() -> None:
    design = one_hot(["a", "a"], ["a", "b"])

    result = category_means(design, np.array([4.0, 6.0]))

    np.testing.assert_allclose(result, [5.0, 0.0])


def test_category_means_returns_an_empty_vector_for_a_zero_width_design() -> None:
    result = category_means(np.empty((3, 0)), np.array([1.0, 2.0, 3.0]))

    assert result.shape == (0,)


def test_category_means_rejects_a_non_two_dimensional_design() -> None:
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        category_means(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_category_means_rejects_a_response_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match response length"):
        category_means(np.ones((3, 2)), np.array([1.0, 2.0]))


def test_additive_effects_are_category_means_minus_the_baseline() -> None:
    design = one_hot(["a", "b"], ["a", "b"])

    result = additive_effects(design, np.array([2.0, 6.0]), baseline=4.0)

    np.testing.assert_allclose(result, [-2.0, 2.0])


def test_additive_effects_are_zero_for_unobserved_columns() -> None:
    design = one_hot(["a", "a"], ["a", "b"])

    result = additive_effects(design, np.array([2.0, 6.0]), baseline=4.0)

    np.testing.assert_allclose(result, [0.0, 0.0])


def test_additive_effects_falls_back_to_category_means_for_a_zero_width_design() -> None:
    result = additive_effects(np.empty((2, 0)), np.array([1.0, 2.0]), baseline=1.5)

    assert result.shape == (0,)


def test_predict_with_effects_adds_the_selected_effects_to_the_baseline() -> None:
    design = one_hot(["a", "b", "a"], ["a", "b"])

    result = predict_with_effects(design, np.array([-1.0, 2.0]), baseline=5.0)

    np.testing.assert_allclose(result, [4.0, 7.0, 4.0])


def test_predict_with_effects_returns_the_baseline_for_a_zero_width_design() -> None:
    result = predict_with_effects(np.empty((3, 0)), np.empty(0), baseline=2.5)

    np.testing.assert_allclose(result, [2.5, 2.5, 2.5])


def test_predict_with_effects_rejects_a_non_two_dimensional_design() -> None:
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        predict_with_effects(np.array([1.0, 0.0]), np.array([1.0, 2.0]), baseline=0.0)


def test_predict_with_effects_rejects_an_effect_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must match design columns"):
        predict_with_effects(np.ones((2, 3)), np.array([1.0, 2.0]), baseline=0.0)


def test_state_float_vector_flattens_nested_lists() -> None:
    result = state_float_vector({"effects": [[1.0], [2.0]]}, "effects")

    assert result is not None
    np.testing.assert_allclose(result, [1.0, 2.0])
    assert result.dtype == np.float64


def test_state_float_vector_returns_none_when_absent() -> None:
    assert state_float_vector({}, "effects") is None


def test_state_float_matrix_promotes_a_flat_list_to_a_column() -> None:
    result = state_float_matrix({"table": [1.0, 2.0, 3.0]}, "table")

    assert result is not None
    assert result.shape == (3, 1)


def test_state_float_matrix_preserves_two_dimensional_payloads() -> None:
    result = state_float_matrix({"table": [[1.0, 2.0], [3.0, 4.0]]}, "table")

    assert result is not None
    assert result.shape == (2, 2)
    assert result.dtype == np.float64


def test_state_float_matrix_returns_none_when_absent() -> None:
    assert state_float_matrix({"table": None}, "table") is None

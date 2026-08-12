"""Tests for the dense-matrix helpers shared by featurizers.

Mirrors :mod:`drevalpy.components.featurizers._matrix`.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._matrix import (
    entity_index_map,
    feature_names_for_view,
    stack_pair_features,
    stack_view_matrix,
    unique_entity_ids,
)
from tests.conftest import MockFeatureSource


def _source() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0])},
            "cl2": {"gene_expression": np.array([3.0, 4.0])},
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )


def test_unique_entity_ids_keeps_first_seen_order() -> None:
    ids = np.array(["cl2", "cl1", "cl2", "cl3"], dtype=str)

    assert unique_entity_ids(ids).tolist() == ["cl2", "cl1", "cl3"]


def test_unique_entity_ids_on_an_empty_array() -> None:
    assert unique_entity_ids(np.array([], dtype=str)).tolist() == []


def test_entity_index_map_maps_ids_to_row_positions() -> None:
    ids = np.array(["cl1", "cl2", "cl3"], dtype=str)

    assert entity_index_map(ids) == {"cl1": 0, "cl2": 1, "cl3": 2}


def test_entity_index_map_keeps_the_last_position_for_duplicates() -> None:
    ids = np.array(["cl1", "cl1"], dtype=str)

    assert entity_index_map(ids) == {"cl1": 1}


def test_feature_names_for_view_delegates_to_the_source() -> None:
    assert feature_names_for_view(_source(), "gene_expression") == ("g1", "g2")


def test_feature_names_for_view_returns_none_for_an_unannotated_view() -> None:
    assert feature_names_for_view(_source(), "mutations") is None


def test_stack_view_matrix_delegates_to_the_source() -> None:
    matrix = stack_view_matrix(_source(), "gene_expression", np.array(["cl2", "cl1"], dtype=str))

    np.testing.assert_allclose(matrix, [[3.0, 4.0], [1.0, 2.0]])


def test_stack_pair_features_concatenates_both_sides() -> None:
    cell_lines = np.array([[1.0, 2.0], [3.0, 4.0]])
    drugs = np.array([[5.0], [6.0]])

    pairs = stack_pair_features(cell_lines, drugs, np.array([0, 1, 1]), np.array([1, 0, 1]))

    np.testing.assert_allclose(pairs, [[1.0, 2.0, 6.0], [3.0, 4.0, 5.0], [3.0, 4.0, 6.0]])


def test_stack_pair_features_short_circuits_an_empty_cell_line_side() -> None:
    drugs = np.array([[5.0], [6.0]])

    pairs = stack_pair_features(np.empty((0, 0)), drugs, np.array([], dtype=int), np.array([1, 0]))

    np.testing.assert_allclose(pairs, [[6.0], [5.0]])


def test_stack_pair_features_short_circuits_an_empty_drug_side() -> None:
    cell_lines = np.array([[1.0, 2.0], [3.0, 4.0]])

    pairs = stack_pair_features(cell_lines, np.empty((0, 0)), np.array([1, 0]), np.array([], dtype=int))

    np.testing.assert_allclose(pairs, [[3.0, 4.0], [1.0, 2.0]])


@pytest.mark.parametrize(
    ("n_cell_line_features", "n_drug_features", "expected_width"),
    [
        pytest.param(2, 3, 5, id="both-sides"),
        pytest.param(1, 1, 2, id="single-column-each"),
    ],
)
def test_stack_pair_features_width_is_the_sum_of_both_sides(
    n_cell_line_features: int,
    n_drug_features: int,
    expected_width: int,
) -> None:
    cell_lines = np.ones((2, n_cell_line_features))
    drugs = np.ones((2, n_drug_features))

    pairs = stack_pair_features(cell_lines, drugs, np.array([0, 1]), np.array([0, 1]))

    assert pairs.shape == (2, expected_width)

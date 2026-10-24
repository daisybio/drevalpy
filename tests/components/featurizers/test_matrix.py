"""Tests for the dense-matrix helpers shared by featurizers (``_matrix``)."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers._matrix import (
    feature_names_for_view,
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
    assert unique_entity_ids(np.array(["cl2", "cl1", "cl2", "cl3"], dtype=str)).tolist() == ["cl2", "cl1", "cl3"]
    assert unique_entity_ids(np.array([], dtype=str)).tolist() == []


def test_view_helpers_delegate_to_the_source() -> None:
    assert feature_names_for_view(_source(), "gene_expression") == ("g1", "g2")
    assert feature_names_for_view(_source(), "mutations") is None
    matrix = stack_view_matrix(_source(), "gene_expression", np.array(["cl2", "cl1"], dtype=str))
    np.testing.assert_allclose(matrix, [[3.0, 4.0], [1.0, 2.0]])

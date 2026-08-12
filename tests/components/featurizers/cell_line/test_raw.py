"""Tests for raw cell-line featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.raw import RawCellLineFeaturizer
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "mutations": np.array([0.0, 1.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "mutations": np.array([1.0, 0.0], dtype=np.float32),
            },
        }
    )


def test_raw_passes_through_view() -> None:
    features = _make_features()
    featurizer = RawCellLineFeaturizer(view="gene_expression")
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix[0], [1.0, 2.0])
    assert np.allclose(matrix[1], [3.0, 4.0])


def test_raw_requires_explicit_view() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        RawCellLineFeaturizer(view="")


def test_raw_prefers_a_precomputed_variant_over_the_view() -> None:
    from drevalpy.components.featurizers.cell_line.raw import RawCellLineFeaturizer as Raw
    from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source

    source = precomputed_source(Raw)
    featurizer = Raw(view="gene_expression")

    featurizer.fit(source, entity_ids=source.identifiers)
    matrix = featurizer.transform(source, source.identifiers)

    assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(matrix, PRECOMPUTED)

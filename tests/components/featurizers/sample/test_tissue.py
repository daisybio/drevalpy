"""Tests for the one-hot tissue sample featurizer; the encoder itself is covered by ``test_one_hot.py``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.tissue import TissueFeaturizer
from drevalpy.types.data.batch.feature_block import BlockSpec
from tests.conftest import MockFeatureSource


def _tissue(**per_sample) -> MockFeatureSource:
    return MockFeatureSource(
        features={cl: {key: np.array([value]) for key, value in view.items()} for cl, view in per_sample.items()}
    )


_IDS = np.array(["cl1", "cl2"], dtype=str)


def test_tissue_one_hot_blocks_and_state_round_trip() -> None:
    features = _tissue(cl1={"tissue": "lung"}, cl2={"tissue": "skin"})
    featurizer = TissueFeaturizer()
    featurizer.fit(features, entity_ids=_IDS)

    blocks = featurizer.transform_blocks(features, _IDS)

    assert featurizer.transform(features, _IDS).shape == (2, 2)
    assert list(blocks["tissue_categories"].values) == ["lung", "skin"]
    assert TissueFeaturizer.output_block_specs == (BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX),)
    restored = TissueFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, _IDS), featurizer.transform(features, _IDS))


def test_tissue_strict_mode_raises_on_missing_annotations() -> None:
    features = _tissue(cl1={"tissue": "lung"}, cl2={"gene_expression": 1.0})
    with pytest.raises(ValueError, match="requires tissue"):
        TissueFeaturizer(allow_missing=False).fit(features, entity_ids=_IDS)
    fitted = TissueFeaturizer(allow_missing=False).fit(features, entity_ids=_IDS[:1])
    with pytest.raises(ValueError, match="requires tissue"):
        fitted.transform(features, _IDS)
    with pytest.raises(ValueError, match="requires tissue"):
        TissueFeaturizer(allow_missing=False).fit(MockFeatureSource(features={}), entity_ids=np.array([], dtype=str))


@pytest.mark.parametrize(
    ("per_sample", "width"),
    [
        ({"cl1": {"tissue": "lung"}, "cl2": {"gene_expression": 1.0}}, 1),
        ({"cl1": {"tissue": "lung"}, "cl2": {"tissue": np.nan}}, 1),
        ({"cl1": {"gene_expression": 1.0}, "cl2": {"gene_expression": 2.0}}, 0),
    ],
    ids=["partial", "nan-label", "fully-absent"],
)
def test_tissue_allow_missing_gives_zero_rows(per_sample, width) -> None:
    features = _tissue(**per_sample)
    featurizer = TissueFeaturizer(allow_missing=True)
    featurizer.fit(features, entity_ids=_IDS)

    matrix = featurizer.transform(features, _IDS)

    assert matrix.shape == (2, width)
    assert featurizer.output_dim == width
    if width:
        np.testing.assert_array_equal(matrix[:, 0], [1.0, 0.0])

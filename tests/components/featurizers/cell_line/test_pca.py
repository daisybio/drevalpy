"""Tests for PCA cell-line featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.pca import PCACellLineFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def _make_features() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {"gene_expression": np.arange(6, dtype=np.float32)},
            "cl2": {"gene_expression": np.arange(6, 12, dtype=np.float32)},
            "cl3": {"gene_expression": np.arange(12, 18, dtype=np.float32)},
        }
    )


def test_pca_reduces_view_dimension() -> None:
    features = _make_features()
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    entity_ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (3, 2)
    assert featurizer.output_dim == 2


def test_pca_requires_explicit_view() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        PCACellLineFeaturizer(view="")


def test_pca_state_roundtrip() -> None:
    features = _make_features()
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    entity_ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    restored = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    restored.set_state(featurizer.get_state())
    assert np.allclose(
        featurizer.transform(features, entity_ids),
        restored.transform(features, entity_ids),
    )


def test_pca_aligns_cross_study_features_by_name() -> None:
    training = FeatureDataset(
        features={
            "cl1": {"methylation": np.array([1.0, 2.0, 3.0])},
            "cl2": {"methylation": np.array([4.0, 5.0, 6.0])},
        },
        meta_info={"methylation": np.array(["a", "b", "c"])},
    )
    cross_study = FeatureDataset(
        features={"cl3": {"methylation": np.array([20.0, 30.0, 40.0])}},
        meta_info={"methylation": np.array(["b", "c", "d"])},
    )
    expected = FeatureDataset(
        features={"cl3": {"methylation": np.array([0.0, 20.0, 30.0])}},
        meta_info={"methylation": np.array(["a", "b", "c"])},
    )
    featurizer = PCACellLineFeaturizer(view="methylation", n_components=2)
    featurizer.fit(training, entity_ids=np.array(["cl1", "cl2"]))

    assert np.allclose(
        featurizer.transform(cross_study, np.array(["cl3"])),
        featurizer.transform(expected, np.array(["cl3"])),
    )

"""Tests for the concatFeaturizers drug featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.concat import ConcatFeaturizersDrugFeaturizer
from drevalpy.models.config import FeaturizerConfig
from drevalpy.registry._builtins import register_builtin_components
from tests.conftest import MockFeatureSource


def _drug_feature_dataset() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "drug1": {
                "morgan_fingerprint": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            },
            "drug2": {
                "morgan_fingerprint": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            },
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2", "fp3"]},
    )


def test_drug_concat_featurizers_fit_transform_and_blocks() -> None:
    register_builtin_components()
    featurizer = ConcatFeaturizersDrugFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="identity", registry="drug"),
        ],
    )
    features = _drug_feature_dataset()
    ids = np.array(["drug1", "drug2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert matrix.shape == (2, 5)
    assert set(blocks) == {"fingerprints", "identity", "identity_categories"}
    assert blocks["fingerprints"].values.shape == (2, 3)
    assert blocks["identity"].values.shape == (2, 2)
    np.testing.assert_allclose(
        matrix,
        np.concatenate([blocks["fingerprints"].values, blocks["identity"].values], axis=1),
    )


def test_concat_rejects_non_numeric_children() -> None:
    register_builtin_components()
    featurizer = ConcatFeaturizersDrugFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="drugGraph", registry="drug"),
        ],
    )
    features = _drug_feature_dataset()
    with pytest.raises(ValueError, match="only numeric_matrix children are supported"):
        featurizer.fit(features, entity_ids=np.array(["drug1"], dtype=str))

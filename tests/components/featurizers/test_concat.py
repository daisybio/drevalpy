"""Tests for concatFeaturizers cell-line and drug featurizers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.config import FeaturizerConfig
from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
from drevalpy.components.featurizers.drug.concat import ConcatFeaturizersDrugFeaturizer
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.datasets.dataset import FeatureDataset


def _feature_dataset() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "mutations": np.array([0.0, 1.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "mutations": np.array([1.0, 0.0], dtype=np.float32),
            },
        },
        meta_info={
            "gene_expression": ["g1", "g2"],
            "mutations": ["m1", "m2"],
        },
    )


def test_concat_featurizers_fit_transform_and_blocks() -> None:
    register_builtin_components()
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            FeaturizerConfig(name="geneExpression", registry="cell_line"),
            FeaturizerConfig(name="mutations", registry="cell_line"),
        ],
    )
    features = _feature_dataset()
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert matrix.shape == (2, 4)
    assert set(blocks) == {"geneExpression", "mutations"}
    assert blocks["geneExpression"].shape == (2, 2)
    assert blocks["mutations"].shape == (2, 2)
    assert np.allclose(matrix, np.concatenate([blocks["geneExpression"], blocks["mutations"]], axis=1))


def _drug_feature_dataset() -> FeatureDataset:
    return FeatureDataset(
        features={
            "drug1": {
                "fingerprints": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            },
            "drug2": {
                "fingerprints": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            },
        },
        meta_info={"fingerprints": ["fp1", "fp2", "fp3"]},
    )


def test_drug_concat_featurizers_fit_transform_and_blocks() -> None:
    register_builtin_components()
    featurizer = ConcatFeaturizersDrugFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="oneHot", registry="drug"),
        ],
    )
    features = _drug_feature_dataset()
    ids = np.array(["drug1", "drug2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert matrix.shape == (2, 5)
    assert set(blocks) == {"fingerprints", "oneHot"}
    assert blocks["fingerprints"].shape == (2, 3)
    assert blocks["oneHot"].shape == (2, 2)
    assert np.allclose(matrix, np.concatenate([blocks["fingerprints"], blocks["oneHot"]], axis=1))

"""Tests for concatFeaturizers cell-line and drug featurizers."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.featurizers.cell_line.concat import (
    ConcatFeaturizersCellLineFeaturizer,
)
from drevalpy.components.featurizers.drug.concat import ConcatFeaturizersDrugFeaturizer
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config import CellLineFeaturizerConfig, FeaturizerConfig
from tests.conftest import MockFeatureSource


def _feature_dataset() -> MockFeatureSource:
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
        },
        meta_info={
            "gene_expression": ["g1", "g2"],
            "mutations": ["m1", "m2"],
        },
    )


def _multi_view_feature_dataset() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "proteomics": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "proteomics": np.array([40.0, 50.0, 60.0], dtype=np.float32),
            },
        }
    )


def test_concat_featurizers_fit_transform_and_blocks() -> None:
    register_builtin_components()
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            FeaturizerConfig(name="raw", view="mutations", registry="cell_line"),
        ],
    )
    features = _feature_dataset()
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert matrix.shape == (2, 4)
    assert set(blocks) == {"gene_expression", "mutations"}
    assert all(isinstance(block, FeatureBlock) for block in blocks.values())
    assert blocks["gene_expression"].values.shape == (2, 2)
    assert blocks["mutations"].values.shape == (2, 2)
    assert blocks["gene_expression"].feature_names == ("g1", "g2")
    np.testing.assert_allclose(
        matrix,
        np.concatenate(
            [blocks["gene_expression"].values, blocks["mutations"].values],
            axis=1,
        ),
    )


def test_concat_uses_canonical_block_names_for_same_name_different_views() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            CellLineFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 1}),
            CellLineFeaturizerConfig(name="pca", view="proteomics", options={"n_components": 1}),
        ],
    )
    features = _multi_view_feature_dataset()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    blocks = featurizer.transform_blocks(features, entity_ids)
    assert set(blocks) == {"gene_expression", "proteomics"}
    assert featurizer.block_dims == {"pca[expression]": 1, "pca[proteomics]": 1}
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)


def test_concat_rejects_duplicate_emitted_block_names() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        ],
    )
    features = _feature_dataset()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    with pytest.raises(ValueError, match="Duplicate featurizer block name 'gene_expression'"):
        featurizer.transform_blocks(features, entity_ids)


def _drug_feature_dataset() -> MockFeatureSource:
    return MockFeatureSource(
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


def test_concat_duplicate_same_name_view_raises() -> None:
    register_builtin_components()
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[expression\\]'"):
        ConcatFeaturizersCellLineFeaturizer(
            featurizers=[
                FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
                FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            ],
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

"""Tests for the shared concatenating featurizer (``shared/concat``)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers.shared.concat import (
    DrugConcatFeaturizer,
    SampleConcatFeaturizer,
    SharedConcatFeaturizer,
)
from drevalpy.models.config import FeaturizerConfig, SampleFeaturizerConfig
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.types.data.batch.feature_block import FeatureBlock
from tests.conftest import MockFeatureSource

_IDS = np.array(["cl1", "cl2"], dtype=str)


def _sample_features(second_view: str = "mutations", width: int = 2) -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                second_view: np.arange(width, dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                second_view: np.arange(width, dtype=np.float32)[::-1].copy(),
            },
        },
        meta_info={"gene_expression": ["g1", "g2"], "mutations": ["m1", "m2"]},
    )


def _drug_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "drug1": {"morgan_fingerprint": np.array([1.0, 0.0, 1.0], dtype=np.float32)},
            "drug2": {"morgan_fingerprint": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2", "fp3"]},
    )


def test_sample_concat_fit_transform_and_blocks() -> None:
    featurizer = SampleConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="sample"),
            FeaturizerConfig(name="raw", view="mutations", registry="sample"),
        ],
    )
    features = _sample_features()
    featurizer.fit(features, entity_ids=_IDS)

    matrix = featurizer.transform(features, _IDS)
    blocks = featurizer.transform_blocks(features, _IDS)

    assert matrix.shape == (2, 4)
    assert set(blocks) == {"gene_expression", "mutations"}
    assert all(isinstance(block, FeatureBlock) for block in blocks.values())
    assert blocks["gene_expression"].feature_names == ("g1", "g2")
    np.testing.assert_allclose(
        matrix, np.concatenate([blocks["gene_expression"].values, blocks["mutations"].values], axis=1)
    )


def test_drug_concat_fit_transform_and_blocks_resolve_against_the_drug_registry() -> None:
    featurizer = DrugConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="identity", registry="drug"),
        ],
    )
    features = _drug_features()
    ids = np.array(["drug1", "drug2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert featurizer._registry == "drug"
    assert matrix.shape == (2, 5)
    assert set(blocks) == {"fingerprints", "identity", "identity_categories"}
    np.testing.assert_allclose(
        matrix, np.concatenate([blocks["fingerprints"].values, blocks["identity"].values], axis=1)
    )


def test_concat_uses_canonical_block_names_for_same_name_different_views() -> None:
    featurizer = SampleConcatFeaturizer(
        featurizers=[
            SampleFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 1}),
            SampleFeaturizerConfig(name="pca", view="proteomics", options={"n_components": 1}),
        ],
    )
    features = _sample_features("proteomics", 3)
    featurizer.fit(features, entity_ids=_IDS)

    assert set(featurizer.transform_blocks(features, _IDS)) == {"gene_expression", "proteomics"}
    assert featurizer.block_dims == {"pca[gene_expression]": 1, "pca[proteomics]": 1}
    assert featurizer.transform(features, _IDS).shape == (2, 2)


def test_concat_rejects_duplicate_block_names_and_selectors() -> None:
    featurizer = SampleConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="sample"),
            FeaturizerConfig(name="scaledGeneExpression", registry="sample"),
        ],
    )
    features = _sample_features()
    featurizer.fit(features, entity_ids=_IDS)
    with pytest.raises(ValueError, match="Duplicate featurizer block name 'gene_expression'"):
        featurizer.transform_blocks(features, _IDS)

    duplicate = FeaturizerConfig(name="raw", view="gene_expression", registry="sample")
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[gene_expression\\]'"):
        SampleConcatFeaturizer(featurizers=[duplicate, duplicate])


def test_concat_rejects_non_numeric_children() -> None:
    featurizer = DrugConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="drugGraph", registry="drug"),
        ],
    )
    with pytest.raises(ValueError, match="only numeric_matrix children are supported"):
        featurizer.fit(_drug_features(), entity_ids=np.array(["drug1"], dtype=str))


def test_concat_registers_one_class_per_side() -> None:
    assert get_sample_featurizer("concatFeaturizers") is SampleConcatFeaturizer
    assert get_drug_featurizer("concatFeaturizers") is DrugConcatFeaturizer
    assert (SampleConcatFeaturizer.side, DrugConcatFeaturizer.side) == ("sample", "drug")
    assert issubclass(SampleConcatFeaturizer, SharedConcatFeaturizer)

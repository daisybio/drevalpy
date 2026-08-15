"""Tests for the shared concatenating featurizer.

Mirrors :mod:`drevalpy.components.featurizers.shared.concat`. The concatenation
logic lives in ``ConcatFeaturizersMixin`` and is covered in
``tests/components/featurizers/test_concat.py``; this file exercises the two side
bindings end to end, including that each resolves its children against its own
registry.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers.shared.concat import (
    CellLineConcatFeaturizer,
    DrugConcatFeaturizer,
    SharedConcatFeaturizer,
)
from drevalpy.models.config import CellLineFeaturizerConfig, FeaturizerConfig
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.types.data.batch.feature_block import FeatureBlock
from tests.conftest import MockFeatureSource


def _cell_line_features() -> MockFeatureSource:
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
        meta_info={"gene_expression": ["g1", "g2"], "mutations": ["m1", "m2"]},
    )


def _multi_view_features() -> MockFeatureSource:
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


def _drug_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "drug1": {"morgan_fingerprint": np.array([1.0, 0.0, 1.0], dtype=np.float32)},
            "drug2": {"morgan_fingerprint": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
        },
        meta_info={"morgan_fingerprint": ["fp1", "fp2", "fp3"]},
    )


def test_cell_line_concat_fit_transform_and_blocks() -> None:
    featurizer = CellLineConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            FeaturizerConfig(name="raw", view="mutations", registry="cell_line"),
        ],
    )
    features = _cell_line_features()
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)
    blocks = featurizer.transform_blocks(features, ids)

    assert matrix.shape == (2, 4)
    assert set(blocks) == {"gene_expression", "mutations"}
    assert all(isinstance(block, FeatureBlock) for block in blocks.values())
    assert blocks["gene_expression"].feature_names == ("g1", "g2")
    np.testing.assert_allclose(
        matrix,
        np.concatenate([blocks["gene_expression"].values, blocks["mutations"].values], axis=1),
    )


def test_drug_concat_fit_transform_and_blocks() -> None:
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

    assert matrix.shape == (2, 5)
    assert set(blocks) == {"fingerprints", "identity", "identity_categories"}
    np.testing.assert_allclose(
        matrix,
        np.concatenate([blocks["fingerprints"].values, blocks["identity"].values], axis=1),
    )


def test_concat_uses_canonical_block_names_for_same_name_different_views() -> None:
    featurizer = CellLineConcatFeaturizer(
        featurizers=[
            CellLineFeaturizerConfig(name="pca", view="gene_expression", options={"n_components": 1}),
            CellLineFeaturizerConfig(name="pca", view="proteomics", options={"n_components": 1}),
        ],
    )
    features = _multi_view_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert set(blocks) == {"gene_expression", "proteomics"}
    assert featurizer.block_dims == {"pca[gene_expression]": 1, "pca[proteomics]": 1}
    assert featurizer.transform(features, entity_ids).shape == (2, 2)


def test_concat_rejects_duplicate_emitted_block_names() -> None:
    featurizer = CellLineConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            FeaturizerConfig(name="scaledGeneExpression", registry="cell_line"),
        ],
    )
    features = _cell_line_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    with pytest.raises(ValueError, match="Duplicate featurizer block name 'gene_expression'"):
        featurizer.transform_blocks(features, entity_ids)


def test_concat_duplicate_same_name_view_raises() -> None:
    with pytest.raises(ValidationError, match="Duplicate featurizer selector 'raw\\[gene_expression\\]'"):
        CellLineConcatFeaturizer(
            featurizers=[
                FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
                FeaturizerConfig(name="raw", view="gene_expression", registry="cell_line"),
            ],
        )


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
    assert get_cell_line_featurizer("concatFeaturizers") is CellLineConcatFeaturizer
    assert get_drug_featurizer("concatFeaturizers") is DrugConcatFeaturizer
    assert CellLineConcatFeaturizer is not DrugConcatFeaturizer


def test_concat_side_is_stamped_per_binding() -> None:
    assert CellLineConcatFeaturizer.side == "cell_line"
    assert DrugConcatFeaturizer.side == "drug"
    assert issubclass(CellLineConcatFeaturizer, SharedConcatFeaturizer)


def test_concat_resolves_children_against_its_own_side_registry() -> None:
    featurizer = DrugConcatFeaturizer(featurizers=[FeaturizerConfig(name="identity", registry="drug")])

    assert featurizer._registry == "drug"

"""Tests for the shared concat, constant and identity featurizers and the mixins behind them."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.sample.raw import RawSampleFeaturizer
from drevalpy.components.featurizers.shared.concat import (
    DrugConcatFeaturizer,
    SampleConcatFeaturizer,
    SharedConcatFeaturizer,
)
from drevalpy.components.featurizers.shared.constant import (
    DrugConstantFeaturizer,
    SampleConstantFeaturizer,
    SharedConstantFeaturizer,
)
from drevalpy.components.featurizers.shared.identity import (
    DrugIdentityFeaturizer,
    SampleIdentityFeaturizer,
    SharedIdentityFeaturizer,
)
from drevalpy.models.config import FeaturizerConfig, SampleFeaturizerConfig
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import metadata as sample_metadata
from drevalpy.types.data.batch.feature_block import FeatureBlock
from tests.conftest import MockFeatureSource

_IDS = np.array(["cl1", "cl2"], dtype=str)
_SIDE_CLASSES = {
    "concat": (SampleConcatFeaturizer, DrugConcatFeaturizer, SharedConcatFeaturizer),
    "constant": (SampleConstantFeaturizer, DrugConstantFeaturizer, SharedConstantFeaturizer),
    "identity": (SampleIdentityFeaturizer, DrugIdentityFeaturizer, SharedIdentityFeaturizer),
}


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


def _raw_configs() -> list[SampleFeaturizerConfig]:
    return [SampleFeaturizerConfig(name="raw", view=view) for view in ("gene_expression", "mutations")]


@pytest.mark.parametrize(
    ("name", "kind"), [("concatFeaturizers", "concat"), ("constant", "constant"), ("identity", "identity")]
)
def test_each_shared_featurizer_registers_one_class_per_side(name: str, kind: str) -> None:
    sample_cls, drug_cls, shared_cls = _SIDE_CLASSES[kind]
    assert get_sample_featurizer(name) is sample_cls
    assert get_drug_featurizer(name) is drug_cls
    assert (sample_cls.side, drug_cls.side, shared_cls.side) == ("sample", "drug", "")
    assert issubclass(sample_cls, shared_cls) and issubclass(drug_cls, shared_cls)


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


def test_concat_rejects_non_numeric_children_and_empty_lists() -> None:
    featurizer = DrugConcatFeaturizer(
        featurizers=[
            FeaturizerConfig(name="fingerprints", registry="drug"),
            FeaturizerConfig(name="drugGraph", registry="drug"),
        ],
    )
    with pytest.raises(ValueError, match="only numeric_matrix children are supported"):
        featurizer.fit(_drug_features(), entity_ids=np.array(["drug1"], dtype=str))
    with pytest.raises(ValueError, match="non-empty list"):
        SampleConcatFeaturizer(featurizers=[])


def test_concat_accepts_pre_built_featurizer_instances() -> None:
    featurizer = SampleConcatFeaturizer(
        featurizers=[RawSampleFeaturizer(view="gene_expression"), RawSampleFeaturizer(view="mutations")]
    )
    featurizer.fit(_sample_features(), entity_ids=_IDS)

    assert featurizer.output_dim == 4
    assert set(featurizer.block_dims) == {"raw[gene_expression]", "raw[mutations]"}


def test_concat_transform_before_fit_raises() -> None:
    featurizer = SampleConcatFeaturizer(featurizers=_raw_configs()[:1])
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer.transform_blocks(_sample_features(), _IDS[:1])


def test_concat_round_trips_state_into_a_fresh_instance() -> None:
    features = _sample_features()
    featurizer = SampleConcatFeaturizer(featurizers=_raw_configs()).fit(features, entity_ids=_IDS)
    restored = SampleConcatFeaturizer(featurizers=_raw_configs())
    restored.set_state(featurizer.get_state())

    assert restored.output_dim == featurizer.output_dim
    assert restored.block_dims == featurizer.block_dims
    np.testing.assert_allclose(restored.transform(features, _IDS), featurizer.transform(features, _IDS))


def test_concat_set_state_ignores_unrelated_keys_and_empty_children_materialize_to_nothing() -> None:
    featurizer = SampleConcatFeaturizer(featurizers=_raw_configs()[:1])
    featurizer.set_state({"child_states": "not-a-dict", "block_dims": None, "output_dim": "seven"})
    assert featurizer.output_dim == 0
    assert featurizer.block_dims == {}
    mixin = ConcatFeaturizersMixin.__new__(ConcatFeaturizersMixin)
    mixin._children = []
    mixin._child_configs = []
    mixin._materialize_children()
    assert mixin._children == []


@pytest.mark.parametrize("featurizer_cls", [SampleConstantFeaturizer, DrugConstantFeaturizer], ids=["sample", "drug"])
def test_constant_is_a_ones_column_with_a_no_op_state(featurizer_cls: type) -> None:
    featurizer = featurizer_cls()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e1", "e2", "e3"], dtype=str)
    assert featurizer.output_dim == 1
    featurizer.fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)
    blocks = featurizer.transform_blocks(features, entity_ids)

    assert matrix.shape == (3, 1)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, 1.0)
    assert list(blocks) == ["constant"]
    assert issubclass(featurizer_cls, (ConstantFeaturizerMixin, SharedConstantFeaturizer))
    assert featurizer_cls.entity_id_only is True
    restored = featurizer_cls()
    restored.set_state(featurizer.get_state())
    assert featurizer.get_state() == {}
    np.testing.assert_allclose(restored.transform(features, entity_ids), matrix)


def test_constant_descriptions_are_worded_per_side() -> None:
    assert "sample" in sample_metadata("constant")["description"]
    assert "drug" in drug_metadata("constant")["description"]


@pytest.mark.parametrize("featurizer_cls", [SampleIdentityFeaturizer, DrugIdentityFeaturizer], ids=["sample", "drug"])
def test_identity_one_hot_blocks_and_state_round_trip(featurizer_cls: type) -> None:
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e2", "e1", "e2"], dtype=str)
    featurizer = featurizer_cls().fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)
    blocks = featurizer.transform_blocks(features, entity_ids)

    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, [[0, 1], [1, 0], [0, 1]])
    assert set(blocks) == {"identity", "identity_categories"}
    assert list(blocks["identity_categories"].values) == ["e1", "e2"]
    restored = featurizer_cls()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 2
    np.testing.assert_allclose(restored.transform(features, entity_ids), matrix)

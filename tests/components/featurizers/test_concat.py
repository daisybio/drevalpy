"""Tests for the shared concat featurizer mixin (``_concat``)."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
from drevalpy.components.featurizers.sample.raw import RawSampleFeaturizer
from drevalpy.components.featurizers.shared.concat import SampleConcatFeaturizer
from drevalpy.models.config import SampleFeaturizerConfig
from tests.conftest import MockFeatureSource

_IDS = np.array(["cl1", "cl2"], dtype=str)


def _sample_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32), "mutations": np.array([0.0, 1.0])},
            "cl2": {"gene_expression": np.array([3.0, 4.0], dtype=np.float32), "mutations": np.array([1.0, 0.0])},
        }
    )


def _raw_configs() -> list[SampleFeaturizerConfig]:
    return [SampleFeaturizerConfig(name="raw", view=view) for view in ("gene_expression", "mutations")]


def test_concat_rejects_non_numeric_children_and_empty_lists() -> None:
    mixin = ConcatFeaturizersMixin.__new__(ConcatFeaturizersMixin)
    mixin._children = [("drugGraph", DrugGraphFeaturizer())]
    with pytest.raises(ValueError, match="only numeric_matrix"):
        mixin._reject_non_numeric_children(mixin._children)
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


def test_concat_set_state_ignores_unrelated_keys() -> None:
    featurizer = SampleConcatFeaturizer(featurizers=_raw_configs()[:1])
    featurizer.set_state({"child_states": "not-a-dict", "block_dims": None, "output_dim": "seven"})
    assert featurizer.output_dim == 0
    assert featurizer.block_dims == {}


def test_concat_refuses_standalone_view_resolution_and_empty_materialization() -> None:
    with pytest.raises(TypeError, match="has no input views of its own"):
        SampleConcatFeaturizer.resolve_input_views()
    mixin = ConcatFeaturizersMixin.__new__(ConcatFeaturizersMixin)
    mixin._children = []
    mixin._child_configs = []
    mixin._materialize_children()
    assert mixin._children == []

"""Tests for the shared concat featurizer mixin.

Mirrors :mod:`drevalpy.components.featurizers._concat`. The two registered
wrappers around this mixin have their own tests in ``cell_line/test_concat.py``
and ``drug/test_concat.py``; this file exercises the mixin's own guards and its
state round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
from drevalpy.components.featurizers.cell_line.raw import RawCellLineFeaturizer
from drevalpy.components.featurizers.drug.drug_graph import DrugGraphFeaturizer
from drevalpy.models.config import CellLineFeaturizerConfig
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
        }
    )


def test_concat_rejects_non_numeric_children() -> None:
    mixin = ConcatFeaturizersMixin.__new__(ConcatFeaturizersMixin)
    mixin._children = [("drugGraph", DrugGraphFeaturizer())]

    with pytest.raises(ValueError, match="only numeric_matrix"):
        mixin._reject_non_numeric_children(mixin._children)


def test_concat_rejects_an_empty_featurizer_list() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        ConcatFeaturizersCellLineFeaturizer(featurizers=[])


def test_concat_accepts_pre_built_featurizer_instances() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            RawCellLineFeaturizer(view="gene_expression"),
            RawCellLineFeaturizer(view="mutations"),
        ],
    )
    features = _cell_line_features()
    ids = np.array(["cl1", "cl2"], dtype=str)

    featurizer.fit(features, entity_ids=ids)

    assert featurizer.output_dim == 4
    assert set(featurizer.block_dims) == {"raw[gene_expression]", "raw[mutations]"}


def test_concat_transform_before_fit_raises() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[CellLineFeaturizerConfig(name="raw", view="gene_expression")],
    )

    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer.transform_blocks(_cell_line_features(), np.array(["cl1"], dtype=str))


def test_concat_round_trips_state_into_a_fresh_instance() -> None:
    features = _cell_line_features()
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            CellLineFeaturizerConfig(name="raw", view="gene_expression"),
            CellLineFeaturizerConfig(name="raw", view="mutations"),
        ],
    ).fit(features, entity_ids=ids)

    restored = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            CellLineFeaturizerConfig(name="raw", view="gene_expression"),
            CellLineFeaturizerConfig(name="raw", view="mutations"),
        ],
    )
    restored.set_state(featurizer.get_state())

    assert restored.output_dim == featurizer.output_dim
    assert restored.block_dims == featurizer.block_dims
    np.testing.assert_allclose(
        restored.transform(features, ids),
        featurizer.transform(features, ids),
    )


def test_concat_set_state_ignores_unrelated_keys() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[CellLineFeaturizerConfig(name="raw", view="gene_expression")],
    )

    featurizer.set_state({"child_states": "not-a-dict", "block_dims": None, "output_dim": "seven"})

    assert featurizer.output_dim == 0
    assert featurizer.block_dims == {}


def test_concat_mixin_refuses_standalone_view_resolution() -> None:
    with pytest.raises(TypeError, match="has no input views of its own"):
        ConcatFeaturizersCellLineFeaturizer.resolve_input_views()


def test_materialize_children_is_a_no_op_without_children_or_configs() -> None:
    mixin = ConcatFeaturizersMixin.__new__(ConcatFeaturizersMixin)
    mixin._children = []
    mixin._child_configs = []

    mixin._materialize_children()

    assert mixin._children == []

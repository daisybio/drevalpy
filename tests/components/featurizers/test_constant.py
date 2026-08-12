"""Tests for the shared constant (intercept) featurizer mixin.

Mirrors :mod:`drevalpy.components.featurizers._constant`. The two registered
wrappers are covered in ``cell_line/test_constant.py`` and
``drug/test_constant.py``; this file pins the mixin's own behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.cell_line.constant import CellLineConstantFeaturizer
from drevalpy.components.featurizers.drug.constant import DrugConstantFeaturizer
from tests.conftest import MockFeatureSource


@pytest.mark.parametrize(
    "featurizer_cls",
    [
        pytest.param(CellLineConstantFeaturizer, id="cell-line"),
        pytest.param(DrugConstantFeaturizer, id="drug"),
    ],
)
def test_constant_featurizers_use_the_shared_mixin(featurizer_cls: type) -> None:
    assert issubclass(featurizer_cls, ConstantFeaturizerMixin)
    assert featurizer_cls.entity_id_only is True


def test_constant_needs_no_source_views() -> None:
    featurizer = CellLineConstantFeaturizer()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)

    featurizer.fit(MockFeatureSource(features={}), entity_ids=entity_ids)

    assert featurizer.output_dim == 1


def test_constant_output_dim_is_one_before_fit() -> None:
    assert CellLineConstantFeaturizer().output_dim == 1


def test_constant_state_round_trip_is_a_no_op() -> None:
    featurizer = CellLineConstantFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    restored = CellLineConstantFeaturizer()
    restored.set_state(featurizer.get_state())

    assert featurizer.get_state() == {}
    np.testing.assert_allclose(
        restored.transform(features, entity_ids),
        featurizer.transform(features, entity_ids),
    )


def test_constant_blocks_are_named_constant() -> None:
    featurizer = DrugConstantFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["d1", "d2", "d3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert list(blocks) == ["constant"]
    np.testing.assert_allclose(blocks["constant"].values, np.ones((3, 1), dtype=np.float32))

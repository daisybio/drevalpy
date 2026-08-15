"""Tests for the shared constant (intercept) featurizer.

Mirrors :mod:`drevalpy.components.featurizers.shared.constant`. The transform logic
itself lives in ``ConstantFeaturizerMixin`` and is covered in
``tests/components/featurizers/test_constant.py``; this file pins the two side
bindings and their registrations.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.shared.constant import (
    CellLineConstantFeaturizer,
    DrugConstantFeaturizer,
    SharedConstantFeaturizer,
)
from drevalpy.registry.cell_line_featurizer import get as get_cell_line_featurizer
from drevalpy.registry.cell_line_featurizer import metadata as cell_line_metadata
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from tests.conftest import MockFeatureSource

_SIDE_CLASSES = [
    pytest.param(CellLineConstantFeaturizer, id="cell-line"),
    pytest.param(DrugConstantFeaturizer, id="drug"),
]


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES)
def test_constant_is_ones_column(featurizer_cls: type) -> None:
    featurizer = featurizer_cls()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e1", "e2", "e3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)

    assert matrix.shape == (3, 1)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, 1.0)
    assert featurizer.output_dim == 1


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES)
def test_constant_uses_the_shared_mixin(featurizer_cls: type) -> None:
    assert issubclass(featurizer_cls, ConstantFeaturizerMixin)
    assert issubclass(featurizer_cls, SharedConstantFeaturizer)
    assert featurizer_cls.entity_id_only is True


def test_constant_registers_one_class_per_side() -> None:
    assert get_cell_line_featurizer("constant") is CellLineConstantFeaturizer
    assert get_drug_featurizer("constant") is DrugConstantFeaturizer
    assert CellLineConstantFeaturizer is not DrugConstantFeaturizer


def test_constant_side_is_stamped_per_binding() -> None:
    assert CellLineConstantFeaturizer.side == "cell_line"
    assert DrugConstantFeaturizer.side == "drug"


def test_constant_descriptions_are_worded_per_side() -> None:
    assert "cell-line" in cell_line_metadata("constant")["description"]
    assert "drug" in drug_metadata("constant")["description"]

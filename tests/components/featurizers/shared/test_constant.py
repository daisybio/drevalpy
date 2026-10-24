"""Tests for the shared constant (intercept) featurizer (``shared/constant``)."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.shared.constant import (
    DrugConstantFeaturizer,
    SampleConstantFeaturizer,
    SharedConstantFeaturizer,
)
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import metadata as drug_metadata
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import metadata as sample_metadata
from tests.conftest import MockFeatureSource


@pytest.mark.parametrize("featurizer_cls", [SampleConstantFeaturizer, DrugConstantFeaturizer], ids=["sample", "drug"])
def test_constant_is_ones_column(featurizer_cls: type) -> None:
    featurizer = featurizer_cls()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e1", "e2", "e3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)

    assert matrix.shape == (3, 1)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, 1.0)
    assert issubclass(featurizer_cls, (ConstantFeaturizerMixin, SharedConstantFeaturizer))
    assert featurizer_cls.entity_id_only is True


def test_constant_registers_one_class_per_side_with_side_worded_descriptions() -> None:
    assert get_sample_featurizer("constant") is SampleConstantFeaturizer
    assert get_drug_featurizer("constant") is DrugConstantFeaturizer
    assert (SampleConstantFeaturizer.side, DrugConstantFeaturizer.side) == ("sample", "drug")
    assert "sample" in sample_metadata("constant")["description"]
    assert "drug" in drug_metadata("constant")["description"]

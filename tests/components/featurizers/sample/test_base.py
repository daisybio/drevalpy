"""Tests for the sample featurizer base class (``sample/base``)."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer, SampleFeaturizer
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import list as list_sample_featurizers


def test_sample_featurizer_is_abstract_and_extends_the_shared_base() -> None:
    assert issubclass(SampleFeaturizer, Featurizer)
    with pytest.raises(TypeError, match="abstract"):
        SampleFeaturizer()


def test_every_registered_sample_featurizer_derives_from_the_base() -> None:
    names = list_sample_featurizers()
    assert names
    assert all(issubclass(get_sample_featurizer(name), SampleFeaturizer) for name in names)


def test_dense_view_binding_resolves_the_side_base_before_the_shared_featurizer() -> None:
    mro = DenseViewSampleFeaturizer.__mro__
    assert issubclass(DenseViewSampleFeaturizer, DenseViewFeaturizer)
    assert mro.index(SampleFeaturizer) < mro.index(Featurizer)

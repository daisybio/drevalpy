"""Tests for featurizer input-view declaration validation at registration time."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcreteSampleFeaturizer(SampleFeaturizer):
    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_features(self, source, entity_ids):
        values = np.zeros((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {"probe": numeric_feature_block(values)})

    @property
    def output_dim(self) -> int:
        return 1


def _register(name: str, cls: type) -> type:
    return register_sample_featurizer(name, description=name, contract=FeatureFormat.NUMERIC_MATRIX)(cls)


def test_featurizer_registration_requires_declared_input_views() -> None:
    with pytest.raises(ValueError, match="does not declare its input views"):
        _register("undeclaredViews", type("UndeclaredViews", (_ConcreteSampleFeaturizer,), {}))

    _register("declaredViews", type("DeclaredViews", (_ConcreteSampleFeaturizer,), {"input_views": ("methylation",)}))

    assert get_sample_featurizer("declaredViews").resolve_input_views() == ("methylation",)


def test_a_featurizer_with_unimplemented_abstract_methods_is_rejected() -> None:
    abstract = type(
        "AbstractFeaturizer",
        (SampleFeaturizer,),
        {"input_views": ("methylation",), "output_dim": property(lambda self: 0)},
    )

    with pytest.raises(ValueError, match=r"does not implement _fit, _transform_features"):
        _register("abstractFeaturizer", abstract)

"""Tests for featurizer input-view declaration validation at registration time."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    register as register_cell_line_featurizer,
)
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


class _ConcreteCellLineFeaturizer(CellLineFeaturizer):
    """Minimal concrete featurizer, so registration is not rejected as abstract."""

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_blocks(self, source, entity_ids):
        return {"probe": numeric_feature_block(np.zeros((len(entity_ids), 1), dtype=np.float32))}

    @property
    def output_dim(self) -> int:
        return 1


def test_featurizer_registration_requires_declared_input_views() -> None:
    with pytest.raises(ValueError, match="does not declare its input views"):

        @register_cell_line_featurizer(
            "undeclaredViews",
            description="no input views declared",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class UndeclaredViews(_ConcreteCellLineFeaturizer):
            pass


def test_declared_input_views_allow_registration() -> None:
    @register_cell_line_featurizer(
        "declaredViews",
        description="declares its input views",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DeclaredViews(_ConcreteCellLineFeaturizer):
        input_views = ("methylation",)

    assert get_cell_line_featurizer("declaredViews").resolve_input_views() == ("methylation",)


def test_a_featurizer_with_unimplemented_abstract_methods_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"does not implement _fit, _transform_blocks"):

        @register_cell_line_featurizer(
            "abstractFeaturizer",
            description="forgot the subclass hooks",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class AbstractFeaturizer(CellLineFeaturizer):
            input_views = ("methylation",)

            @property
            def output_dim(self) -> int:
                return 0

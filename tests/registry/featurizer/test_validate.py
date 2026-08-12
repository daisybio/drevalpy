"""Tests for featurizer input-view declaration validation at registration time."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    register as register_cell_line_featurizer,
)
from tests.registry._helpers import isolated_component_registries


@pytest.fixture(autouse=True)
def _clear_registries() -> Iterator[None]:
    yield from isolated_component_registries()


def test_featurizer_registration_requires_declared_input_views() -> None:
    from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer

    with pytest.raises(ValueError, match="does not declare its input views"):

        @register_cell_line_featurizer(
            "undeclaredViews",
            description="no input views declared",
            contract=FeatureFormat.NUMERIC_MATRIX,
        )
        class UndeclaredViews(CellLineFeaturizer):
            def fit(self, features, *, entity_ids=None, context=None):
                return self

            def transform(self, features, entity_ids):
                raise NotImplementedError

            @property
            def output_dim(self) -> int:
                return 0


def test_declared_input_views_allow_registration() -> None:
    from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer

    @register_cell_line_featurizer(
        "declaredViews",
        description="declares its input views",
        contract=FeatureFormat.NUMERIC_MATRIX,
    )
    class DeclaredViews(CellLineFeaturizer):
        input_views = ("methylation",)

        def fit(self, features, *, entity_ids=None, context=None):
            return self

        def transform(self, features, entity_ids):
            raise NotImplementedError

        @property
        def output_dim(self) -> int:
            return 0

    assert get_cell_line_featurizer("declaredViews").resolve_input_views() == ("methylation",)

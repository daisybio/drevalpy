"""Single-view drug featurizer."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.base import DenseViewDrugFeaturizer
from drevalpy.registry.drug_featurizer import register


@register(
    "view",
    description="Pass through one dense drug view from a FeatureSource.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ViewDrugFeaturizer(DenseViewDrugFeaturizer):
    """Featurize one drug view without additional transformation."""

    input_views: ClassVar[tuple[str, ...]] = ("morgan_fingerprint",)

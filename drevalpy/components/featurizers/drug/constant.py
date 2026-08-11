"""Drug constant (one-category / intercept) featurizer."""

from __future__ import annotations

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.registry.drug_featurizer import register


@register(
    "constant",
    description="Constant one-column intercept features with no drug identity.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class DrugConstantFeaturizer(ConstantFeaturizerMixin, DrugFeaturizer):
    """Emit ones for every drug entity."""

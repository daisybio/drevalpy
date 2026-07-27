"""Drug constant (one-category / intercept) featurizer."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "constant",
    description="Constant one-column intercept features with no drug identity.",
    category="native",
    contract=FeatureKind.DENSE,
)
class DrugConstantFeaturizer(ConstantFeaturizerMixin, DrugFeaturizer):
    """Emit ones for every drug entity."""

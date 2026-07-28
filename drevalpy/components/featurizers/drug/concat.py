"""Concatenate outputs from multiple drug featurizers."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer


@register_drug_featurizer(
    "concatFeaturizers",
    description="Concatenate dense outputs from multiple drug featurizers.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ConcatFeaturizersDrugFeaturizer(ConcatFeaturizersMixin, DrugFeaturizer):
    """Fit child featurizers independently and concatenate their dense outputs."""

    _not_fitted_msg = "ConcatFeaturizersDrugFeaturizer must be fit before transform"

    def __init__(
        self,
        *,
        featurizers: list[Any] | None = None,
    ) -> None:
        self._init_concat(featurizers=featurizers, registry="drug")

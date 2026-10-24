"""Concatenating featurizer, shared by both entity sides."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers._side_binding import register_for_sides
from drevalpy.components.featurizers.base import Featurizer


@register_for_sides(
    "concatFeaturizers",
    description={
        "sample": "Concatenate dense outputs from multiple sample featurizers.",
        "drug": "Concatenate dense outputs from multiple drug featurizers.",
    },
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SharedConcatFeaturizer(ConcatFeaturizersMixin, Featurizer):
    """Fit child featurizers independently and concatenate their dense outputs."""

    def __init__(
        self,
        *,
        featurizers: list[Any] | None = None,
    ) -> None:
        """Initialize instance state.

        Children are resolved against the registry of this binding's own side, which
        registration stamped onto the class.

        :param featurizers: Child featurizers, either built instances or config mappings; must be non-empty.
        """
        self._init_concat(featurizers=featurizers, registry=self.side)

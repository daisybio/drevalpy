"""Generic raw dense pass-through featurizer for one omics view."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.registry.sample_featurizer import register


@register(
    "raw",
    description="Pass through one dense omics view without preprocessing.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class RawSampleFeaturizer(DenseViewSampleFeaturizer):
    """Featurize one omics view as a dense matrix without transformation."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str) -> None:
        """Bind the omics view to pass through.

        :param view: Name of the omics view to emit unchanged.
        :raises ValueError: If *view* is empty or blank.
        """
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        super().__init__(view=view)

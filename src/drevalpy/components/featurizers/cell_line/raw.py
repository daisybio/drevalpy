"""Generic raw dense pass-through featurizer for one omics view."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register


@register(
    "raw",
    description="Pass through one dense omics view without preprocessing.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class RawCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Featurize one omics view as a dense matrix without transformation."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str) -> None:
        """Initialize instance state.

        :param view: view.
        :raises ValueError: Raised on invalid input.
        """
        if not view or not view.strip():
            msg = "raw featurizer requires an explicit view"
            raise ValueError(msg)
        super().__init__(view=view)

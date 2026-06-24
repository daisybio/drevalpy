"""Pathway featurizer for Precily."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizers.cell_line.view import ViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "pathways",
    description="Precomputed GSVA pathway features for Precily.",
    category="general_purpose",
)
class PathwaysCellLineFeaturizer(ViewCellLineFeaturizer):
    """Pathways cell line featurizer component."""

    output_contract: ClassVar[FeatureContract] = FeatureContract(kind=FeatureKind.DENSE, view="pathways")

    def __init__(self, *, view: str = "pathways") -> None:
        super().__init__(view=view)

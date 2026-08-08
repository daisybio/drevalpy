"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BionicCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Bionic cell line featurizer component."""

    input_views: ClassVar[tuple[str, ...]] = ("bionic_features",)

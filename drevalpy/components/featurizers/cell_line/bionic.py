"""BIONIC featurizer for DIPK."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers.cell_line.omics.dense_view import DenseViewCellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "bionic",
    description="Precomputed BIONIC cell-line features for DIPK.",
    category="general_purpose",
    contract=FeatureKind.DENSE,
)
class BionicCellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Bionic cell line featurizer component."""

    _default_view = "bionic_features"

"""Base classes for cell-line featurizers."""

from __future__ import annotations

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer


class CellLineFeaturizer(Featurizer):
    """Base for featurizers that read one or more cell-line feature views."""


class DenseViewCellLineFeaturizer(DenseViewFeaturizer, CellLineFeaturizer):
    """Cell-line binding of the shared single-view dense featurizer base."""

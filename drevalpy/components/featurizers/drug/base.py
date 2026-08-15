"""Base classes for drug featurizers."""

from __future__ import annotations

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer


class DrugFeaturizer(Featurizer):
    """Base for featurizers that read drug feature views."""


class DenseViewDrugFeaturizer(DenseViewFeaturizer, DrugFeaturizer):
    """Drug binding of the shared single-view dense featurizer base."""

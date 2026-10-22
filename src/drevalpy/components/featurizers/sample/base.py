"""Base classes for sample featurizers."""

from __future__ import annotations

from drevalpy.components.featurizers._dense_view import DenseViewFeaturizer
from drevalpy.components.featurizers.base import Featurizer


class SampleFeaturizer(Featurizer):
    """Base for featurizers that read one or more sample views."""


class DenseViewSampleFeaturizer(DenseViewFeaturizer, SampleFeaturizer):
    """Sample binding of the shared single-view dense featurizer base."""

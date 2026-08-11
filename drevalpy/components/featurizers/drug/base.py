"""Base class for drug featurizers."""

from __future__ import annotations

from drevalpy.components.featurizers.base import Featurizer


class DrugFeaturizer(Featurizer):
    """Base for featurizers that read drug feature views."""

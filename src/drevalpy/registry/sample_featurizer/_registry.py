"""Sample featurizer registry singleton and class."""

from __future__ import annotations

from drevalpy.registry.featurizer._base import FeaturizerRegistry


class SampleFeaturizerRegistry(FeaturizerRegistry):
    """Registry for sample featurizers."""

    def __init__(self) -> None:
        """Initialize with fixed sample identity."""
        super().__init__("sample_featurizer", "Sample featurizer", "sample_featurizers", side="sample")


sample_featurizer_registry = SampleFeaturizerRegistry()

"""Drug featurizer registry singleton and class."""

from __future__ import annotations

from drevalpy.registry.featurizer._base import FeaturizerRegistry


class DrugFeaturizerRegistry(FeaturizerRegistry):
    """Registry for drug featurizers."""

    def __init__(self) -> None:
        """Initialize with fixed drug identity."""
        super().__init__("drug_featurizer", "Drug featurizer", "drug_featurizers", side="drug")


drug_featurizer_registry = DrugFeaturizerRegistry()

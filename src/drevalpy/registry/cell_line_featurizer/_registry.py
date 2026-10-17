"""Cell-line featurizer registry singleton and class."""

from __future__ import annotations

from drevalpy.registry.featurizer._base import FeaturizerRegistry


class CellLineFeaturizerRegistry(FeaturizerRegistry):
    """Registry for cell-line featurizers."""

    def __init__(self) -> None:
        """Initialize with fixed cell-line identity."""
        super().__init__("cell_line_featurizer", "Cell line featurizer", "cell_line_featurizers", side="cell_line")


cell_line_featurizer_registry = CellLineFeaturizerRegistry()

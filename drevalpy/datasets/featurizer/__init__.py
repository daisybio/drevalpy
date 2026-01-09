"""Featurizers for converting drug and cell line data to embeddings.

This module provides abstract base classes and concrete implementations for
featurizing drugs and cell lines for drug response prediction models.

Drug Featurizers:
    - DrugFeaturizer: Abstract base class for drug featurizers
    - ChemBERTaFeaturizer: ChemBERTa transformer embeddings from SMILES
    - DrugGraphFeaturizer: Molecular graph representations
    - MolGNetFeaturizer: MolGNet graph neural network embeddings

Cell Line Featurizers:
    - CellLineFeaturizer: Abstract base class for cell line featurizers
    - PCAFeaturizer: PCA dimensionality reduction for omics data

Example usage::

    from drevalpy.datasets.featurizer import ChemBERTaFeaturizer, PCAFeaturizer

    # Drug features
    drug_featurizer = ChemBERTaFeaturizer(device="cuda")
    drug_features = drug_featurizer.load_or_generate("data", "GDSC1")

    # Cell line features
    cell_featurizer = PCAFeaturizer(n_components=100)
    cell_features = cell_featurizer.load_or_generate("data", "GDSC1")
"""

# Cell line featurizers
from .cell_line import (
    CellLineFeaturizer,
    PCAFeaturizer,
)

# Drug featurizers
from .drug import (
    ChemBERTaFeaturizer,
    DrugFeaturizer,
    DrugGraphFeaturizer,
    MolGNetFeaturizer,
)

__all__ = [
    # Drug featurizers
    "DrugFeaturizer",
    "ChemBERTaFeaturizer",
    "DrugGraphFeaturizer",
    "MolGNetFeaturizer",
    # Cell line featurizers
    "CellLineFeaturizer",
    "PCAFeaturizer",
]

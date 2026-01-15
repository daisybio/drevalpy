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

Mixins for DRP Models:
    - ChemBERTaMixin: Provides load_drug_features using ChemBERTa
    - DrugGraphMixin: Provides load_drug_features using DrugGraphFeaturizer
    - MolGNetMixin: Provides load_drug_features using MolGNet
    - PCAMixin: Provides load_cell_line_features using PCA

Example usage::

    from drevalpy.datasets.featurizer import ChemBERTaFeaturizer, PCAFeaturizer

    # Drug features
    drug_featurizer = ChemBERTaFeaturizer(device="cuda")
    drug_features = drug_featurizer.load_or_generate("data", "GDSC1")

    # Cell line features
    cell_featurizer = PCAFeaturizer(n_components=100)
    cell_features = cell_featurizer.load_or_generate("data", "GDSC1")

Example using mixins in a model::

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.featurizer import ChemBERTaMixin, PCAMixin

    class MyModel(ChemBERTaMixin, PCAMixin, DRPModel):
        # ChemBERTaMixin provides load_drug_features
        # PCAMixin provides load_cell_line_features
        ...
"""

# Cell line featurizers
from .cell_line import (
    CellLineFeaturizer,
    PCAFeaturizer,
    PCAMixin,
)

# Drug featurizers
from .drug import (
    ChemBERTaFeaturizer,
    ChemBERTaMixin,
    DrugFeaturizer,
    DrugGraphFeaturizer,
    DrugGraphMixin,
    MolGNetFeaturizer,
    MolGNetMixin,
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
    # Mixins
    "ChemBERTaMixin",
    "DrugGraphMixin",
    "MolGNetMixin",
    "PCAMixin",
]

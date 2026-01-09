"""Drug graph featurizer for converting SMILES to molecular graphs."""

import argparse
import os
from pathlib import Path

import torch
from torch_geometric.data import Data

from drevalpy.datasets.dataset import FeatureDataset

from .base import DrugFeaturizer

try:
    from rdkit import Chem
except ImportError:
    Chem = None


# Atom feature configuration
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "hybridization": [],  # Will be populated after rdkit import check
}

# Bond feature configuration
BOND_FEATURES = {
    "bond_type": [],  # Will be populated after rdkit import check
}


def _init_rdkit_features():
    """Initialize RDKit-dependent feature configurations.

    :raises ImportError: If rdkit package is not installed
    """
    if Chem is None:
        raise ImportError("Please install rdkit package for drug graphs featurizer: pip install rdkit")

    ATOM_FEATURES["hybridization"] = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    BOND_FEATURES["bond_type"] = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]


def _one_hot_encode(value, choices):
    """Create a one-hot encoding for a value in a list of choices.

    :param value: The value to be one-hot encoded.
    :param choices: A list of possible choices for the value.
    :return: A list representing the one-hot encoding.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


class DrugGraphFeaturizer(DrugFeaturizer):
    """Featurizer that converts SMILES strings to molecular graphs.

    The graphs are stored as torch_geometric.data.Data objects with:
        - x: Node features (atom features)
        - edge_index: Edge connectivity
        - edge_attr: Edge features (bond features)

    Example usage::

        featurizer = DrugGraphFeaturizer()
        features = featurizer.load_or_generate("data", "GDSC1")
    """

    def __init__(self, device: str = "cpu"):
        """Initialize the drug graph featurizer.

        :param device: Device to use (not used for graph generation, but kept for API consistency)
        """
        super().__init__(device=device)
        _init_rdkit_features()

    def featurize(self, smiles: str) -> Data | None:
        """Convert a SMILES string to a molecular graph.

        :param smiles: SMILES string representing the drug
        :returns: torch_geometric.data.Data object or None if conversion fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_features_list = []
        for atom in mol.GetAtoms():
            features = []
            features.extend(_one_hot_encode(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"]))
            features.extend(_one_hot_encode(atom.GetDegree(), ATOM_FEATURES["degree"]))
            features.extend(_one_hot_encode(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"]))
            features.extend(_one_hot_encode(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"]))
            features.extend(_one_hot_encode(atom.GetHybridization(), ATOM_FEATURES["hybridization"]))
            features.append(atom.GetIsAromatic())
            features.append(atom.IsInRing())
            atom_features_list.append(features)
        x = torch.tensor(atom_features_list, dtype=torch.float)

        # Edge index and edge features
        edge_indices = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Edge features
            features = []
            features.extend(_one_hot_encode(bond.GetBondType(), BOND_FEATURES["bond_type"]))
            features.append(bond.GetIsConjugated())
            features.append(bond.IsInRing())

            edge_indices.extend([[i, j], [j, i]])
            edge_features_list.extend([features, features])  # Same features for both directions

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the feature view name.

        :returns: 'drug_graphs'
        """
        return "drug_graphs"

    @classmethod
    def get_output_filename(cls) -> str:
        """Return the output directory name for cached graphs.

        :returns: 'drug_graphs'
        """
        return "drug_graphs"

    def _save_embeddings(self, embeddings: list, drug_ids: list[str], output_path: Path) -> None:
        """Save graph embeddings to disk as individual .pt files.

        :param embeddings: List of Data objects
        :param drug_ids: List of drug identifiers
        :param output_path: Directory path to save the graphs
        """
        os.makedirs(output_path, exist_ok=True)
        for drug_id, graph in zip(drug_ids, embeddings, strict=True):
            if graph is not None:
                torch.save(graph, output_path / f"{drug_id}.pt")

    def load_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load pre-generated graph embeddings from disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the graph embeddings
        :raises FileNotFoundError: If the graphs directory is not found
        """
        graphs_dir = Path(data_path) / dataset_name / self.get_output_filename()

        if not graphs_dir.exists():
            raise FileNotFoundError(
                f"Graphs directory not found: {graphs_dir}. "
                f"Use load_or_generate() to automatically generate graphs."
            )

        feature_name = self.get_feature_name()
        features = {}

        for graph_file in graphs_dir.glob("*.pt"):
            drug_id = graph_file.stem
            graph = torch.load(graph_file)  # noqa: S614
            features[drug_id] = {feature_name: graph}

        return FeatureDataset(features)

    def load_or_generate(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load cached graphs or generate and cache them if not available.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the drug graphs
        """
        output_path = Path(data_path) / dataset_name / self.get_output_filename()

        if output_path.exists() and any(output_path.glob("*.pt")):
            return self.load_embeddings(data_path, dataset_name)
        else:
            print(f"Graphs not found at {output_path}. Generating...")
            return self.generate_embeddings(data_path, dataset_name)


class DrugGraphMixin:
    """Mixin that provides drug graph loading for DRP models.

    This mixin implements load_drug_features using the DrugGraphFeaturizer.
    It automatically generates graphs if they don't exist.

    Example usage::

        from drevalpy.models.drp_model import DRPModel
        from drevalpy.datasets.featurizer.drug.drug_graph import DrugGraphMixin

        class MyModel(DrugGraphMixin, DRPModel):
            drug_views = ["drug_graphs"]
            ...
    """

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load drug graph features.

        Uses the DrugGraphFeaturizer to load pre-generated graphs or generate
        them automatically if they don't exist.

        :param data_path: Path to the data directory, e.g., 'data/'
        :param dataset_name: Name of the dataset, e.g., 'GDSC1'
        :returns: FeatureDataset containing the drug graphs
        """
        featurizer = DrugGraphFeaturizer()
        return featurizer.load_or_generate(data_path, dataset_name)


def main():
    """Process drug SMILES and save molecular graphs from command line."""
    parser = argparse.ArgumentParser(description="Generate molecular graphs for drugs.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    featurizer = DrugGraphFeaturizer()
    featurizer.generate_embeddings(args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()

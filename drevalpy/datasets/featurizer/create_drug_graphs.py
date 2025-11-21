"""
Preprocesses drug SMILES strings into graph representations.

This script takes a dataset name as input, reads the corresponding
drug_smiles.csv file, and converts each SMILES string into a
torch_geometric.data.Data object. The resulting graph objects are saved
to {data_path}/{dataset_name}/drug_graphs/{drug_name}.pt.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("Please install rdkit package for drug graphs featurizer: pip install rdkit")

# Atom feature configuration
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

# Bond feature configuration
BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
}


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


def _smiles_to_graph(smiles: str):
    """
    Converts a SMILES string to a torch_geometric.data.Data object.

    :param smiles: The SMILES string for the drug.
    :return: A Data object representing the molecular graph, or None if conversion fails.
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


def main():
    """Main function to run the preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess drug SMILES to graphs.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_dir = Path(args.data_path).resolve()
    smiles_file = data_dir / dataset_name / "drug_smiles.csv"
    output_dir = data_dir / dataset_name / "drug_graphs"

    if not smiles_file.exists():
        print(f"Error: {smiles_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    smiles_df = pd.read_csv(smiles_file)

    print(f"Processing {len(smiles_df)} drugs for dataset {dataset_name}...")

    for _, row in tqdm(smiles_df.iterrows(), total=smiles_df.shape[0]):
        drug_id = row["pubchem_id"]
        smiles = row["canonical_smiles"]

        graph = _smiles_to_graph(smiles)

        if graph:
            torch.save(graph, output_dir / f"{drug_id}.pt")

    print(f"Finished processing. Graphs saved to {output_dir}")


if __name__ == "__main__":
    main()

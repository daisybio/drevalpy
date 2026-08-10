"""Preprocesses drug SMILES strings into graph representations.

Reads SMILES from a .h5mu file and writes drug graphs to mdata.uns["drug_graphs"].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as md
import torch
from torch_geometric.data import Data
from tqdm import tqdm

try:
    from rdkit import Chem
except ImportError as err:
    raise ImportError("Please install rdkit package for drug graphs featurizer: pip install rdkit") from err

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

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
}


def _one_hot_encode(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def _smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

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

    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = []
        features.extend(_one_hot_encode(bond.GetBondType(), BOND_FEATURES["bond_type"]))
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        edge_indices.extend([[i, j], [j, i]])
        edge_features_list.extend([features, features])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main(h5mu_path: Path) -> None:
    """Compute drug graphs from SMILES and write to mdata.uns['drug_graphs'].

    :param h5mu_path: Path to the .h5mu file.
    """
    mdata = md.read(str(h5mu_path))
    response = mdata.mod["response"]

    smiles_col = "canonical_smiles"
    if smiles_col not in response.var.columns:
        msg = f"Column {smiles_col!r} not found in response.var. Available: {list(response.var.columns)}"
        raise ValueError(msg)

    drug_graphs: dict[str, Data] = {}

    print(f"Processing {len(response.var_names)} drugs...")
    for drug_id in tqdm(response.var_names):
        smiles = response.var.loc[drug_id, smiles_col]
        if not isinstance(smiles, str) or not smiles:
            continue
        graph = _smiles_to_graph(smiles)
        if graph is not None:
            drug_graphs[drug_id] = graph

    mdata.uns["drug_graphs"] = drug_graphs
    mdata.write(str(h5mu_path))
    print(f"Wrote {len(drug_graphs)} drug graphs to mdata.uns['drug_graphs'] in {h5mu_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute drug graphs from SMILES and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    args = parser.parse_args()
    main(args.h5mu_path)

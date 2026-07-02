r"""
Drug featurizer for the Precily model using SMILESVec embeddings.

Reads the SMILES that DrEvalPy already ships for a dataset and writes a CSV
of drug features keyed by pubchem_id, in the format Precily's
load_drug_features expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def _load_smiles(data_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load drug SMILES from DrEvalPy's expected file structure.

    :param data_path: Root directory containing dataset subfolders.
    :param dataset_name: Name of the dataset (subfolder).
    :return: DataFrame with columns 'pubchem_id' and 'canonical_smiles'.
    :raises FileNotFoundError: If the SMILES file does not exist.
    :raises ValueError: If required columns are missing.
    """
    smiles_file = Path(data_path) / dataset_name / "drug_smiles.csv"
    if not smiles_file.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")
    df = pd.read_csv(smiles_file, dtype=str)
    required_cols = {"pubchem_id", "canonical_smiles"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {required_cols} in {smiles_file}, got {df.columns.tolist()}")
    return df


def _smilesvec(smiles: str, kv: KeyedVectors, k: int = 8, dim: int = 100) -> np.ndarray:
    """
    Convert a SMILES string to a vector using SMILESVec (word2vec on substrings).

    :param smiles: Input SMILES string.
    :param kv: Gensim KeyedVectors model.
    :param k: Length of substrings (chemical words).
    :param dim: Dimensionality of the vectors.
    :return: Mean vector of found substrings, or zero vector if none found.
    """
    if len(smiles) < k:
        words = [smiles]
    else:
        words = [smiles[i : i + k] for i in range(len(smiles) - k + 1)]  # noqa: E203

    vecs = [kv[w] for w in words if w in kv.key_to_index]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def main() -> None:
    """
    Command-line entry point: generate drug features using SMILESVec.

    Reads drug SMILES from a dataset, loads a pre‑trained SMILESVec model,
    computes substring embeddings, and writes a CSV file
    'precily_drug_features.csv' inside the dataset folder.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--smilesvec_model", required=True, help="path to pretrained SMILESVec word2vec model")
    parser.add_argument("--k", type=int, default=8, help="length of substring (chemical word)")
    args = parser.parse_args()

    # Load SMILES
    smiles_df = _load_smiles(args.data_path, args.dataset_name)

    # Load SMILESVec model
    kv = KeyedVectors.load_word2vec_format(args.smilesvec_model, binary=False)

    # Generate features
    rows = []
    n_oov = 0
    for _, row in smiles_df.iterrows():
        vec = _smilesvec(row["canonical_smiles"], kv, k=args.k, dim=100)
        if not np.any(vec):
            n_oov += 1
        rows.append([row["pubchem_id"], *vec.tolist()])

    n_features = 100
    columns = ["pubchem_id"] + [f"smv_{i}" for i in range(n_features)]
    out_df = pd.DataFrame(rows, columns=columns)

    # Write output
    out_path = Path(args.data_path) / args.dataset_name / "drug_smilesvec.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} drugs x {n_features} features -> {out_path}")
    if n_oov:
        print(f"WARNING: {n_oov} drugs produced all-zero (unparsable/OOV) vectors.")


if __name__ == "__main__":
    main()

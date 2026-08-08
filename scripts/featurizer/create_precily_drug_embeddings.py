"""Drug featurizer using SMILESVec embeddings.

Reads SMILES from a .h5mu file, computes SMILESVec embeddings, and writes
the result to response.varm["smilesvec"].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

import mudata as md


def _smilesvec(smiles: str, kv: KeyedVectors, k: int = 8, dim: int = 100) -> np.ndarray:
    if len(smiles) < k:
        words = [smiles]
    else:
        words = [smiles[i : i + k] for i in range(len(smiles) - k + 1)]
    vecs = [kv[w] for w in words if w in kv.key_to_index]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def main(h5mu_path: Path, *, smilesvec_model: str, k: int = 8) -> None:
    """Compute SMILESVec embeddings and write to response.varm['smilesvec'].

    :param h5mu_path: Path to the .h5mu file.
    :param smilesvec_model: Path to pretrained SMILESVec word2vec model.
    :param k: Length of substrings (chemical words).
    """
    mdata = md.read(str(h5mu_path))
    response = mdata.mod["response"]

    smiles_col = "canonical_smiles"
    if smiles_col not in response.var.columns:
        msg = f"Column {smiles_col!r} not found in response.var."
        raise ValueError(msg)

    kv = KeyedVectors.load_word2vec_format(smilesvec_model, binary=False)
    dim = 100

    embeddings = []
    n_oov = 0
    print(f"Processing {len(response.var_names)} drugs...")
    for drug_id in tqdm(response.var_names):
        smiles = response.var.loc[drug_id, smiles_col]
        if not isinstance(smiles, str) or not smiles:
            embeddings.append(np.zeros(dim, dtype=np.float32))
            n_oov += 1
            continue
        vec = _smilesvec(smiles, kv, k=k, dim=dim)
        if not np.any(vec):
            n_oov += 1
        embeddings.append(vec)

    embeddings_array = np.vstack(embeddings).astype(np.float32)
    response.varm["smilesvec"] = embeddings_array

    mdata.write(str(h5mu_path))
    print(f"Wrote SMILESVec embeddings ({embeddings_array.shape}) to response.varm['smilesvec'] in {h5mu_path}")
    if n_oov:
        print(f"WARNING: {n_oov} drugs produced all-zero (unparsable/OOV) vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SMILESVec drug embeddings and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--smilesvec_model", required=True, help="Path to pretrained SMILESVec word2vec model")
    parser.add_argument("--k", type=int, default=8, help="Length of substring (chemical word)")
    args = parser.parse_args()
    main(args.h5mu_path, smilesvec_model=args.smilesvec_model, k=args.k)

"""Preprocesses drug SMILES strings into ChemBERTa embeddings.

Reads SMILES from a .h5mu file and writes the embeddings back as response.varm["chemberta"].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as md
import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as err:
    raise ImportError(
        "Please install transformers package for ChemBERTa embedding featurizer: pip install transformers"
    ) from err

_CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
_CHEMBERTA_REVISION = "761d6a1"


def _smiles_to_chemberta(smiles: str, tokenizer, model, device="cpu") -> np.ndarray:
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


def main(h5mu_path: Path, *, device: str = "cpu") -> None:
    """Compute ChemBERTa embeddings and write to response.varm['chemberta'].

    :param h5mu_path: Path to the .h5mu file.
    :param device: Torch device (cpu or cuda).
    """
    mdata = md.read(str(h5mu_path))
    response = mdata.mod["response"]

    smiles_col = "canonical_smiles"
    if smiles_col not in response.var.columns:
        msg = f"Column {smiles_col!r} not found in response.var. Available: {list(response.var.columns)}"
        raise ValueError(msg)

    tokenizer = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL, revision=_CHEMBERTA_REVISION)
    model_obj = AutoModel.from_pretrained(_CHEMBERTA_MODEL, revision=_CHEMBERTA_REVISION)
    model_obj.to(device)
    model_obj.eval()

    smiles_series = response.var[smiles_col]
    embeddings = []

    print(f"Processing {len(smiles_series)} drugs...")
    for drug_id in tqdm(response.var_names):
        smiles = smiles_series[drug_id]
        embedding = _smiles_to_chemberta(str(smiles), tokenizer, model_obj, device=device)
        embeddings.append(embedding)

    embeddings_array = np.vstack(embeddings).astype(np.float32)
    response.varm["chemberta"] = embeddings_array

    mdata.write(str(h5mu_path))
    print(f"Wrote chemberta embeddings ({embeddings_array.shape}) to response.varm['chemberta'] in {h5mu_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ChemBERTa drug embeddings and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()
    main(args.h5mu_path, device=args.device)

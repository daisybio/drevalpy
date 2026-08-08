"""Preprocesses drug SMILES strings into BPE-encoded embeddings.

WARNING: This featurizer produces problematic embeddings and should ONLY be used
with the PharmaFormer model. It replicates the original PharmaFormer implementation
for compatibility, but the embeddings have known issues and should not be used
for any other models.

Details about the issues are explained in:
https://github.com/daisybio/drevalpy/pull/336#discussion_r2682718948

Reads SMILES from a .h5mu file and writes BPE-encoded SMILES to response.varm["bpe_smiles"].
"""

from __future__ import annotations

import argparse
import codecs
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

import mudata as md

try:
    from subword_nmt.apply_bpe import BPE
    from subword_nmt.learn_bpe import learn_bpe
except ImportError as err:
    raise ImportError("Please install subword-nmt package for BPE SMILES featurizer: pip install subword-nmt") from err


def main(h5mu_path: Path, *, num_symbols: int = 10000, max_length: int = 128) -> None:
    """Compute BPE-encoded SMILES and write to response.varm['bpe_smiles'].

    :param h5mu_path: Path to the .h5mu file.
    :param num_symbols: Number of BPE symbols to learn.
    :param max_length: Maximum length of encoded SMILES (padding/truncation).
    """
    mdata = md.read(str(h5mu_path))
    response = mdata.mod["response"]

    smiles_col = "canonical_smiles"
    if smiles_col not in response.var.columns:
        msg = f"Column {smiles_col!r} not found in response.var. Available: {list(response.var.columns)}"
        raise ValueError(msg)

    smiles_series = response.var[smiles_col].dropna()

    print(f"Learning BPE codes from {len(smiles_series)} SMILES strings...")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp_file:
        tmp_smiles_file = Path(tmp_file.name)
        for smiles in smiles_series:
            tmp_file.write(f"{smiles}\n")

    bpe_codes_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".codes")
    bpe_codes_path = Path(bpe_codes_file.name)
    bpe_codes_file.close()

    try:
        with codecs.open(str(tmp_smiles_file), encoding="utf-8") as f_in:
            with codecs.open(str(bpe_codes_path), "w", encoding="utf-8") as f_out:
                learn_bpe(f_in, f_out, num_symbols=num_symbols)
    finally:
        tmp_smiles_file.unlink(missing_ok=True)

    with codecs.open(str(bpe_codes_path), encoding="utf-8") as f_in:
        bpe = BPE(f_in)
    bpe_codes_path.unlink(missing_ok=True)

    embeddings_list = []
    print(f"Encoding {len(response.var_names)} drugs...")
    for drug_id in tqdm(response.var_names):
        smiles = response.var.loc[drug_id, smiles_col]
        if not isinstance(smiles, str) or not smiles:
            embeddings_list.append(np.zeros(max_length, dtype=np.int32))
            continue
        bpe_processed = bpe.process_line(smiles)
        encoded = [ord(char) for char in bpe_processed]
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded = np.pad(encoded, (0, max_length - len(encoded)), "constant").tolist()
        embeddings_list.append(encoded)

    embeddings_array = np.array(embeddings_list, dtype=np.int32)
    response.varm["bpe_smiles"] = embeddings_array

    mdata.write(str(h5mu_path))
    print(f"Wrote BPE SMILES embeddings ({embeddings_array.shape}) to response.varm['bpe_smiles'] in {h5mu_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BPE-encoded SMILES and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--num-symbols", type=int, default=10000, help="Number of BPE symbols to learn")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum length of encoded SMILES")
    args = parser.parse_args()
    main(args.h5mu_path, num_symbols=args.num_symbols, max_length=args.max_length)

"""Preprocesses drug SMILES strings into BPE-encoded embeddings."""

import argparse
import codecs
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from subword_nmt.apply_bpe import BPE
    from subword_nmt.learn_bpe import learn_bpe
except ImportError:
    raise ImportError("Please install subword-nmt package for BPE SMILES featurizer: pip install subword-nmt")


def create_bpe_smiles_embeddings(
    data_path: str,
    dataset_name: str,
    num_symbols: int = 10000,
    max_length: int = 128,
) -> None:
    """
    Create BPE-encoded SMILES embeddings for drugs.

    1. Read drug_smiles.csv
    2. Learn BPE codes from all SMILES strings
    3. Apply BPE to each SMILES
    4. Convert to character ordinals
    5. Pad/truncate to max_length
    6. Save to drug_bpe_smiles.csv

    :param data_path: Path to the data folder
    :param dataset_name: Name of the dataset to process
    :param num_symbols: Number of BPE symbols to learn
    :param max_length: Maximum length of encoded SMILES (padding/truncation)
    :raises FileNotFoundError: If drug_smiles.csv is not found
    :raises Exception: If a drug fails to process
    """
    data_dir = Path(data_path).resolve()
    dataset_dir = data_dir / dataset_name

    smiles_file = dataset_dir / "drug_smiles.csv"
    bpe_codes_path = dataset_dir / "bpe.codes"
    output_file = dataset_dir / "drug_bpe_smiles.csv"

    if not smiles_file.exists():
        raise FileNotFoundError(f"Error: {smiles_file} not found.")

    # Read SMILES data
    smiles_df = pd.read_csv(smiles_file, dtype={"canonical_smiles": str, "pubchem_id": str})
    smiles_df = smiles_df.dropna(subset=["canonical_smiles"])

    print(f"Learning BPE codes from {len(smiles_df)} SMILES strings...")

    # Create temporary file with SMILES strings for BPE learning
    # learn_bpe expects one item per line
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp_file:
        tmp_smiles_file = tmp_file.name
        for smiles in smiles_df["canonical_smiles"]:
            tmp_file.write(f"{smiles}\n")

    # Learn BPE codes from SMILES corpus
    try:
        with codecs.open(tmp_smiles_file, encoding="utf-8") as f_in:
            with codecs.open(bpe_codes_path, "w", encoding="utf-8") as f_out:
                learn_bpe(f_in, f_out, num_symbols=num_symbols)
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_smiles_file):
            os.remove(tmp_smiles_file)

    print(f"BPE codes saved to {bpe_codes_path}")

    # Load BPE encoder
    with codecs.open(bpe_codes_path, encoding="utf-8") as f_in:
        bpe = BPE(f_in)

    # Encode each SMILES string
    embeddings_list = []
    drug_ids = []

    print(f"Encoding {len(smiles_df)} SMILES strings...")

    for row in tqdm(smiles_df.itertuples(index=False), total=len(smiles_df)):
        drug_id = row.pubchem_id
        smiles = row.canonical_smiles

        try:
            # Apply BPE
            bpe_processed = bpe.process_line(smiles)
            # Convert to character ordinals
            encoded = [ord(char) for char in bpe_processed]
            # Pad/truncate to max_length
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded = np.pad(encoded, (0, max_length - len(encoded)), "constant").tolist()

            embeddings_list.append(encoded)
            drug_ids.append(drug_id)
        except Exception as e:
            print(f"\nFailed to process drug {drug_id} with SMILES: {smiles}")
            print(f"Error: {e}")
            raise e

    # Create DataFrame with pubchem_id and encoded features
    embeddings_df = pd.DataFrame(embeddings_list)
    embeddings_df.columns = [f"feature_{i}" for i in range(max_length)]
    embeddings_df.insert(0, "pubchem_id", drug_ids)
    embeddings_df.to_csv(output_file, index=False)

    print(f"Finished processing. BPE-encoded SMILES saved to {output_file}")


def main():
    """Process drug SMILES and save BPE-encoded embeddings."""
    parser = argparse.ArgumentParser(description="Preprocess drug SMILES to BPE-encoded embeddings.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--num-symbols", type=int, default=10000, help="Number of BPE symbols to learn")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum length of encoded SMILES")
    args = parser.parse_args()

    create_bpe_smiles_embeddings(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        num_symbols=args.num_symbols,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

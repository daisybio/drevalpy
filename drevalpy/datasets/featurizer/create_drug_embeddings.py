"""Preprocesses drug SMILES strings into ChemBERTa embeddings."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Load ChemBERTa
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model.eval()


def _smiles_to_chemberta(smiles: str, device="cpu"):
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

    embedding = hidden_states.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy()


def main():
    """Process drug SMILES and save ChemBERTa embeddings.

    :raises Exception: If a drug fails to process.
    """
    parser = argparse.ArgumentParser(description="Preprocess drug SMILES to ChemBERTa embeddings.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    device = args.device

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_dir = project_root / "data"
    smiles_file = data_dir / dataset_name / "drug_smiles.csv"
    output_file = data_dir / dataset_name / "drug_chemberta_embeddings.csv"

    if not smiles_file.exists():
        print(f"Error: {smiles_file} not found.")
        return

    smiles_df = pd.read_csv(smiles_file, dtype={"canonical_smiles": str, "pubchem_id": str})
    embeddings_list = []
    drug_ids = []

    print(f"Processing {len(smiles_df)} drugs for dataset {dataset_name}...")

    for row in tqdm(smiles_df.itertuples(index=False), total=len(smiles_df)):
        drug_id = row.pubchem_id
        smiles = row.canonical_smiles

        try:
            embedding = _smiles_to_chemberta(smiles, device=device)
            embeddings_list.append(embedding)
            drug_ids.append(drug_id)
        except Exception as e:
            print()
            print(smiles)
            print()
            print(f"Failed to process {drug_id}")
            raise e

    embeddings_array = pd.DataFrame(embeddings_list)
    embeddings_array.insert(0, "drug_id", drug_ids)
    embeddings_array.to_csv(output_file, index=False)

    print(f"Finished processing. Embeddings saved to {output_file}")


if __name__ == "__main__":
    main()

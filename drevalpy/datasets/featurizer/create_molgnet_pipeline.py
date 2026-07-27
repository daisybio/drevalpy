"""Orchestration helpers for the MolGNet embedding featurizer script."""

from __future__ import annotations

import argparse
import os
import pickle  # noqa: S403
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def resolve_molgnet_dataset_dir(data_path: str, dataset_name: str) -> Path:
    """Resolve and validate the dataset directory under ``data_path``."""
    data_dir = Path(data_path).expanduser().resolve()
    dataset_dir = data_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def load_smiles_map(dataset_dir: Path, smiles_col: str, id_col: str) -> dict[Any, str]:
    """Load drug id → SMILES mapping from ``drug_smiles.csv``."""
    smiles_csv = dataset_dir / "drug_smiles.csv"
    if not smiles_csv.exists():
        raise FileNotFoundError(f"Expected SMILES CSV at: {smiles_csv}")
    df = pd.read_csv(smiles_csv)
    if smiles_col not in df.columns or id_col not in df.columns:
        msg = f"Provided columns not in CSV: {smiles_col}, {id_col}"
        raise ValueError(msg)
    df = df.dropna(subset=[smiles_col])
    return dict(zip(df[id_col], df[smiles_col]))


def build_graph_dict(smiles_map: dict[Any, str]) -> dict[Any, Data]:
    """Convert SMILES strings to torch_geometric graph objects."""
    from rdkit import Chem

    from .create_molgnet_embeddings import mol_to_graph_data_obj_complex

    graph_dict: dict[Any, Data] = {}
    failed_conversions: list[tuple[Any, str, str]] = []
    for idx, smi in tqdm(smiles_map.items(), desc="building graphs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_conversions.append((idx, smi, "MolFromSmiles returned None"))
            continue
        try:
            graph_dict[idx] = mol_to_graph_data_obj_complex(mol)
        except Exception as exc:
            failed_conversions.append((idx, smi, str(exc)))
    _report_graph_conversion_failures(failed_conversions)
    return graph_dict


def _report_graph_conversion_failures(failed_conversions: list[tuple[Any, str, str]]) -> None:
    if failed_conversions:
        print(f"\n{len(failed_conversions)} molecules failed to convert to graphs.")
        for idx, smi, err in failed_conversions:
            print(f"Failed to convert {idx} (SMILES: {smi}): {err}")
    else:
        print("\nAll molecules converted to graphs successfully.")


def save_graph_dict(graph_dict: dict[Any, Data], path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(graph_dict, handle)


def resolve_torch_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_molgnet_model():
    from .create_molgnet_embeddings import MolGNet

    return MolGNet(
        num_layer=5,
        emb_dim=768,
        heads=12,
        num_message_passing=3,
        drop_ratio=0.0,
    )


def load_molgnet_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)  # noqa: S614
    try:
        model.load_state_dict(ckpt)
    except Exception:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            raise


def run_molgnet_inference(
    graph_dict: dict[Any, Data],
    model,
    device: torch.device,
) -> dict[Any, torch.Tensor]:
    from .create_molgnet_embeddings import AddSegId, SelfLoop

    self_loop = SelfLoop()
    add_seg = AddSegId()
    molgnet_dict: dict[Any, torch.Tensor] = {}
    with torch.no_grad():
        for idx, graph in tqdm(graph_dict.items(), desc="running model"):
            try:
                prepared = add_seg(self_loop(graph)).to(device)
                emb = model(prepared)
                molgnet_dict[idx] = emb.cpu()
            except Exception as exc:
                print(f"Inference failed for {idx}: {exc}")
    return molgnet_dict


def write_molgnet_drug_csvs(molgnet_dict: dict[Any, torch.Tensor], drugs_dir: Path) -> None:
    from .create_molgnet_embeddings import tensor_to_csv_friendly

    os.makedirs(drugs_dir, exist_ok=True)
    for idx, emb in tqdm(molgnet_dict.items(), desc="writing csvs"):
        arr = tensor_to_csv_friendly(emb)
        df_emb = pd.DataFrame(arr)
        out_path = drugs_dir / f"MolGNet_{idx}.csv"
        df_emb.to_csv(out_path, sep="\t", index=False)


def run_molgnet_pipeline(args: argparse.Namespace) -> None:
    """Execute path resolution through CSV output for MolGNet embeddings."""
    dataset_dir = resolve_molgnet_dataset_dir(args.data_path, args.dataset_name)
    data_dir = dataset_dir.parent
    out_graphs = str(dataset_dir / "GRAPH_dict.pkl")
    out_molg = str(dataset_dir / "MolGNet_dict.pkl")

    smiles_map = load_smiles_map(dataset_dir, args.smiles_col, args.id_col)
    graph_dict = build_graph_dict(smiles_map)
    save_graph_dict(graph_dict, out_graphs)

    device = resolve_torch_device(args.device)
    model = create_molgnet_model()
    load_molgnet_checkpoint(model, data_dir / args.checkpoint, device)
    model = model.to(device)
    model.eval()

    molgnet_dict = run_molgnet_inference(graph_dict, model, device)
    with open(out_molg, "wb") as handle:
        pickle.dump(molgnet_dict, handle)

    out_drugs_dir = dataset_dir / "DIPK_features/Drugs"
    write_molgnet_drug_csvs(molgnet_dict, out_drugs_dir)

    print("Done.")
    print("Graphs saved to:", out_graphs)
    print("Node embeddings saved to:", out_molg)
    print("Per-drug CSVs in:", out_drugs_dir)

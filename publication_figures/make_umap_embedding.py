"""
Create UMAP embeddings for drugs in the CTRPv2 dataset along with a ZINC background.

Saves the resulting 2D embeddings and related metadata to a .npz file at the end.
(For plotting, see make_umap_embedding_plot_highlight.py)
"""

# flake8: noqa: D103
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import umap
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoModel, AutoTokenizer

path_to_source_data_dir = "../results/SourceData"

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["UMAP_DISABLE_NUMBA"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYNN_DESCENT_DISABLE_NUMBA"] = "1"
embed_type = "ecfp"  # "ecfp" or "chemberta"
chemberta_model = "seyonec/ChemBERTa-zinc-base-v1"
chemberta_pooling = "mean"  # "mean" or "cls"
batch_size = 64
max_length = 256

focus_csv = f"{path_to_source_data_dir}/drug_names_CTRPv2.csv"
smiles_csv = f"{path_to_source_data_dir}/all_smiles_final.csv"
zinc_csv = f"{path_to_source_data_dir}/250k_rndm_zinc_drugs_clean_3.csv"
smiles_col = "canonical_smiles"

n_bg_fit = 200000
n_bg_plot = 200000
bg_transform_batch = 8000
rng = np.random.default_rng(0)


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def looks_like_combo(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name_l = name.lower()
    if any(tok in name_l for tok in [" + ", "+", "/", "&", " and ", " with ", ","]):
        return True
    if re.search(r"\bcombo\b|\bcombination\b|\bmixture\b|\bco-?admin", name_l):
        return True
    return False


def largest_fragment_smiles(smi: str) -> str | None:
    if not isinstance(smi, str):
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    frags = sorted(frags, key=lambda mol: mol.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0], canonical=True)


def find_smiles_column(df):
    for c in df.columns:
        if "smile" in c.lower() or "smi" in c.lower():
            return c
    raise ValueError("could not find SMILES column")


def collect_valid_smiles(series, n_target, rng):
    vals = [v for v in series.tolist() if isinstance(v, str)]
    idx = np.arange(len(vals))
    rng.shuffle(idx)
    out = []
    for i, pos in enumerate(idx, 1):
        sm = largest_fragment_smiles(vals[pos])
        if sm:
            out.append(sm)
        if n_target and len(out) >= n_target:
            break
        if i % 5000 == 0:
            log(f"screened {i} rows, collected {len(out)} valid smiles")
    return out[:n_target]


fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fpgen_size = 2048


def morgan_bits(smiles_list):
    X = np.zeros((len(smiles_list), fpgen_size), dtype=np.uint8)
    for i, smi in enumerate(smiles_list, 1):
        m = Chem.MolFromSmiles(smi)
        if m:
            bv = fpgen.GetFingerprint(m)
            on = bv.GetOnBits()
            X[i - 1, list(on)] = 1
        if i % 5000 == 0:
            log(f"fingerprinted {i} molecules")
    return X


def chemberta_embed(smiles_list, model_name, pooling="mean", batch_size=128, max_length=256):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    valid_idx, texts = [], []
    for i, smi in enumerate(smiles_list):
        if isinstance(smi, str) and Chem.MolFromSmiles(smi):
            valid_idx.append(i)
            texts.append(smi)
    if not texts:
        return np.zeros((len(smiles_list), 1)), np.array([], int)
    with torch.no_grad():
        dummy_enc = tok(texts[:1], return_tensors="pt", truncation=True, max_length=max_length)
        emb_dim = mdl(**dummy_enc.to(device)).last_hidden_state.shape[-1]
        Z = np.zeros((len(smiles_list), emb_dim), np.float32)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]  # noqa: E203
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = mdl(**enc)
            if pooling == "cls":
                emb = out.last_hidden_state[:, 0, :]
            else:
                mask = enc.attention_mask.unsqueeze(-1)
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            Z[valid_idx[start : start + batch_size]] = emb.cpu().numpy()  # noqa: E203
            log(f"chemberta embedded {start + len(batch)} of {len(texts)}")
    return Z, np.array(valid_idx, int)


log("loading result tables")
res = pd.read_csv(f"{path_to_source_data_dir}/main_results/true_vs_pred.csv")
res = res[(res["test_mode"] == "LDO") & (res["algorithm"] == "MultiOmicsNeuralNetwork")]
res = res[["drug_name", "CV_split"]]
res["is_train"] = res["CV_split"] != 0

log("loading focus and smiles tables")
src = pd.read_csv(focus_csv, dtype={"pubchem_id": str})
smiles_df = pd.read_csv(smiles_csv, dtype={"pubchem_id": str})
focus = src[~src["drug_name"].map(looks_like_combo)].dropna(subset=["pubchem_id"])
focus["pubchem_id"] = focus["pubchem_id"].str.strip()
focus_merge = (
    focus.merge(smiles_df[["pubchem_id", smiles_col]].rename(columns={smiles_col: "smiles"}), on="pubchem_id")
    .dropna(subset=["smiles"])
    .drop_duplicates(subset=["pubchem_id"])
)
focus_merge["smiles"] = focus_merge["smiles"].map(largest_fragment_smiles)
train_test_info = res.drop_duplicates(subset=["drug_name"])
focus_merge = focus_merge.merge(train_test_info, on="drug_name", how="left").fillna({"is_train": False})

focus_smiles = focus_merge["smiles"].tolist()
focus_is_train = focus_merge["is_train"].values

log("loading ZINC background")
zdf = pd.read_csv(zinc_csv)
zcol = find_smiles_column(zdf)
s_all = zdf[zcol]
smi_bg_fit = collect_valid_smiles(s_all, n_bg_fit, rng)
smi_bg_plot = collect_valid_smiles(s_all, n_bg_plot, rng)

log("embedding fit set")
if embed_type == "ecfp":
    X_bg_fit = morgan_bits(smi_bg_fit)
    X_focus = morgan_bits(focus_smiles)
    metric = "cosine"
else:
    X_bg_fit, _ = chemberta_embed(smi_bg_fit, chemberta_model, chemberta_pooling, batch_size, max_length)
    X_focus, _ = chemberta_embed(focus_smiles, chemberta_model, chemberta_pooling, batch_size, max_length)
    metric = "cosine"

log("fitting UMAP")
um = umap.UMAP(
    n_neighbors=4,
    n_components=2,
    metric=metric,
    n_epochs=100,
    random_state=1,
    force_approximation_algorithm=True,
    low_memory=True,
)
um.fit(X_bg_fit)
log("transforming embeddings")
X_focus_2d = um.transform(X_focus)
X_bg_2d = um.transform(
    morgan_bits(smi_bg_plot)
    if embed_type == "ecfp"
    else chemberta_embed(smi_bg_plot, chemberta_model, chemberta_pooling, batch_size, max_length)[0]
)

np.savez_compressed(
    "LDO_embedding_data.npz",
    X_focus_2d=X_focus_2d,
    X_bg_2d=X_bg_2d,
    focus_is_train=focus_is_train,
    focus_drug_names=focus_merge["drug_name"].values,
    focus_pubchem_ids=focus_merge["pubchem_id"].values,
)
log("saved LDO_embedding_data.npz")

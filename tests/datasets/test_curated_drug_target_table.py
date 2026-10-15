"""Tests for the scripts that build and check the curated drug target table."""

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.featurizer.check_curated_drug_target_table import check_candidate, summarize
from drevalpy.datasets.featurizer.create_curated_drug_target_table import NOT_REPRESENTABLE, SEED_FILE, curate


def _seed(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drug_name": f"drug{i}",
                "pubchem_id": str(i),
                "biomarker_gene": gene,
                "sensitive_status": status,
                "selectivity": "selective",
                "notes": "",
                "origin": origin,
            }
            for i, (gene, status, origin) in enumerate(rows)
        ]
    )


def test_curate_drops_unrepresentable_and_unknown_genes():
    """Fusion drivers and genes outside the matrix are dropped, nothing else."""
    seed = _seed([("BRAF", "activating", "existing"), ("ALK", "activating", "draft"), ("XYZ", "LoF", "draft")])
    curated, report = curate(seed, {"BRAF", "KRAS"})
    assert curated["biomarker_gene"].tolist() == ["BRAF"]
    assert report["outcome"].tolist()[0] == "kept"
    assert report["outcome"].str.startswith(("not representable", "gene not")).tolist() == [False, True, True]


def test_curate_rejects_bad_seed():
    """An unknown status or a duplicate drug is an error."""
    with pytest.raises(ValueError):
        curate(_seed([("BRAF", "gain", "existing")]), {"BRAF"})
    seed = _seed([("BRAF", "activating", "existing"), ("KRAS", "activating", "existing")])
    seed["pubchem_id"] = "1"
    with pytest.raises(ValueError):
        curate(seed, {"BRAF", "KRAS"})


def test_shipped_seed_is_valid():
    """The shipped seed passes the checks."""
    seed = pd.read_csv(SEED_FILE, dtype=str)
    curated, report = curate(seed, set(seed["biomarker_gene"]))
    assert len(curated) == (~seed["biomarker_gene"].isin(NOT_REPRESENTABLE)).sum()
    assert report["outcome"].isin(["kept", "not representable (fusion or amplification)"]).all()


def test_candidate_verdicts():
    """A clear effect is supported, a tiny group is untestable, no difference is no signal."""
    rng = np.random.default_rng(0)
    cells = [f"CVCL_{i}" for i in range(80)]
    status = pd.DataFrame({"G": ["activating"] * 20 + ["wildtype"] * 60}, index=cells)
    auc = np.concatenate([rng.normal(0.4, 0.05, 20), rng.normal(0.8, 0.05, 60)])
    response = pd.DataFrame({"cellosaurus_id": cells, "pubchem_id": "1", "AUC": auc})
    assert check_candidate(response, status, "1", "G", "activating")["verdict"] == "supported"
    assert check_candidate(response, status, "1", "G", "LoF")["verdict"] == "untestable"
    flat = response.assign(AUC=rng.normal(0.6, 0.05, 80))
    assert check_candidate(flat, status, "1", "G", "activating")["verdict"] == "no signal"
    assert check_candidate(response, status, "2", "G", "activating")["verdict"] == "no data"


def test_summarize_priority():
    """One supporting dataset is enough; only testable datasets without signal refute."""
    assert summarize({"a": {"verdict": "no signal"}, "b": {"verdict": "supported"}}) == "supported"
    assert summarize({"a": {"verdict": "no signal"}, "b": {"verdict": "untestable"}}) == "refuted"
    assert summarize({"a": {"verdict": "no data"}}) == "untestable"

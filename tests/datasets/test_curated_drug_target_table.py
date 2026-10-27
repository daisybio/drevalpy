"""Tests for the scripts that build and check the curated drug target table."""

import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from drevalpy.datasets.featurizer import create_curated_drug_target_table as curated_script
from drevalpy.datasets.featurizer.check_curated_drug_target_table import check_candidate, summarize
from drevalpy.datasets.featurizer.create_curated_drug_target_table import curate


def _seed(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drug_name": f"drug{pid}",
                "pubchem_id": pid,
                "biomarker_gene": gene,
                "sensitive_status": status,
                "evidence_type": "civic_predictive_target",
                "evidence_level": level,
                "n_evidence_items": items,
            }
            for pid, gene, status, level, items in rows
        ]
    )


def test_curate_drops_unrepresentable_and_unknown_genes():
    """Fusion drivers and genes outside the matrix are dropped, nothing else."""
    seed = _seed(
        [("1", "BRAF", "activating", "A", "1"), ("2", "ALK", "activating", "A", "1"), ("3", "XYZ", "LoF", "A", "1")]
    )
    curated, report = curate(seed, {"BRAF", "KRAS"})
    assert curated["biomarker_gene"].tolist() == ["BRAF"]
    assert report["outcome"].tolist()[0] == "kept"
    assert report["outcome"].str.startswith(("not representable", "gene not")).tolist() == [False, True, True]


def test_curate_keeps_one_row_per_drug():
    """Representable genes first, then the best evidence level, then the most evidence items, then the seed order."""
    seed = _seed(
        [
            ("1", "ALK", "activating", "A", "9"),  # not representable, however good
            ("1", "KRAS", "activating", "D", "9"),
            ("1", "BRAF", "activating", "B", "1"),  # best level among the representable
            ("2", "KRAS", "activating", "B", "1"),
            ("2", "BRAF", "activating", "B", "5"),  # same level, more items
            ("3", "KRAS", "wildtype", "", ""),  # literature rule without level
            ("3", "BRAF", "LoF", "E", "1"),
            ("4", "KRAS", "activating", "C", "2"),
            ("4", "BRAF", "activating", "C", "2"),  # complete tie: the first row stays
        ]
    )
    curated, report = curate(seed, {"BRAF", "KRAS"})
    assert curated["pubchem_id"].tolist() == ["1", "2", "3", "4"]
    assert curated["biomarker_gene"].tolist() == ["BRAF", "BRAF", "BRAF", "KRAS"]
    assert curated["pubchem_id"].is_unique
    assert report["outcome"].tolist() == [
        "not representable (fusion or amplification)",
        "other row of the drug selected",
        "kept",
        "other row of the drug selected",
        "kept",
        "other row of the drug selected",
        "kept",
        "kept",
        "other row of the drug selected",
    ]


def test_curate_rejects_bad_seed():
    """An unknown status is an error."""
    with pytest.raises(ValueError):
        curate(_seed([("1", "BRAF", "gain", "A", "1")]), {"BRAF"})


def test_missing_seed_is_built(tmp_path, monkeypatch):
    """
    A seed file that does not exist is built with create_target_mutation_seed.py first.

    :param tmp_path: pytest temporary directory
    :param monkeypatch: pytest monkeypatch fixture
    """
    directory = tmp_path / "meta" / "target_mutation"
    directory.mkdir(parents=True)
    pd.DataFrame({"gene": ["BRAF", "KRAS"]}).to_csv(directory / "functional_status_matrix.csv", index=False)
    seed_file = tmp_path / "seed.csv"
    calls = []

    def fake_build(argv):
        calls.append(argv)
        _seed([("1", "BRAF", "activating", "A", "1")]).to_csv(argv[argv.index("--out") + 1], index=False)

    monkeypatch.setattr(curated_script.create_target_mutation_seed, "main", fake_build)
    monkeypatch.setattr(sys, "argv", ["x", "--data_path", str(tmp_path), "--seed", str(seed_file)])
    curated_script.main()
    assert calls == [["--data_path", str(tmp_path), "--out", str(seed_file)]]
    assert pd.read_csv(directory / "curated_drug_target_variant.csv")["biomarker_gene"].tolist() == ["BRAF"]
    curated_script.main()  # the seed exists now, it is not built again
    assert len(calls) == 1


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


def test_candidate_verdicts_underpowered_and_weak():
    """A small sensitive group is underpowered; a small but significant effect is weak."""
    rng = np.random.default_rng(1)
    cells = [f"CVCL_{i}" for i in range(60)]
    status = pd.DataFrame({"G": ["activating"] * 6 + ["wildtype"] * 54}, index=cells)
    flat = pd.DataFrame({"cellosaurus_id": cells, "pubchem_id": "1", "AUC": rng.normal(0.6, 0.05, 60)})
    assert check_candidate(flat, status, "1", "G", "activating")["verdict"] == "underpowered"

    cells = [f"CVCL_{i}" for i in range(2000)]
    status = pd.DataFrame({"G": ["activating"] * 1000 + ["wildtype"] * 1000}, index=cells)
    grid = norm.ppf((np.arange(1000) + 0.5) / 1000) * 0.1  # fixed spread, so that d is exactly -0.2
    auc = np.concatenate([0.58 + grid, 0.6 + grid])
    response = pd.DataFrame({"cellosaurus_id": cells, "pubchem_id": "1", "AUC": auc})
    result = check_candidate(response, status, "1", "G", "activating")
    assert -0.3 < result["d"] < 0 and result["p"] < 0.01
    assert result["verdict"] == "weak"


def test_summarize_priority():
    """One supporting dataset is enough; only testable datasets without signal refute."""
    assert summarize({"a": {"verdict": "no signal"}, "b": {"verdict": "supported"}}) == "supported"
    assert summarize({"a": {"verdict": "no signal"}, "b": {"verdict": "untestable"}}) == "refuted"
    assert summarize({"a": {"verdict": "no data"}}) == "untestable"
    assert summarize({"a": {"verdict": "underpowered"}, "b": {"verdict": "no signal"}}) == "underpowered"
    assert summarize({"a": {"verdict": "weak"}, "b": {"verdict": "no signal"}}) == "weak"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "module",
    [
        "drevalpy.datasets.featurizer.create_target_mutation_status",
        "drevalpy.datasets.featurizer.create_curated_drug_target_table",
        "drevalpy.datasets.featurizer.check_curated_drug_target_table",
    ],
)
def test_scripts_start_as_modules(module, monkeypatch):
    """Each script can be started with ``python -m`` and prints its usage.

    :param module: dotted name of the script
    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(module, run_name="__main__")
    assert exit_info.value.code == 0

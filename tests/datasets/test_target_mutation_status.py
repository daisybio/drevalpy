"""Tests for the scripts that build the functional status matrix and the curated drug target table."""

import hashlib
import sys

import pandas as pd
import pytest

from drevalpy.datasets.featurizer import (
    check_curated_drug_target_table,
    create_curated_drug_target_table,
)
from drevalpy.datasets.featurizer import create_target_mutation_status as status_script

GENES = status_script.GENES


@pytest.mark.parametrize(
    "gene, hotspot, lof, onc, pc, expected",
    [
        ("BRAF", True, False, True, True, "activating"),
        ("BRAF", False, True, False, True, "LoF"),
        ("BRAF", True, True, False, True, "activating"),  # hotspot wins over LikelyLoF
        ("BRAF", False, True, True, False, "activating"),  # splice variant without protein change keeps the flag
        ("BRAF", False, True, False, False, "other"),
        ("BRAF", False, False, False, True, "other"),
        ("TP53", True, False, False, True, "LoF"),
        ("PTEN", False, True, False, False, "LoF"),
        ("NF1", False, False, True, True, "other"),
    ],
)
def test_variant_status(gene, hotspot, lof, onc, pc, expected):
    """
    Rules for one variant, oncogene and tumour suppressor.

    :param expected: expected result
    :param gene: gene symbol
    :param hotspot: variant is a hotspot
    :param lof: variant is likely loss of function
    :param onc: variant is an oncogene high impact variant
    :param pc: variant has a protein change
    """
    assert status_script.variant_status(gene, hotspot, lof, onc, pc) == expected


@pytest.mark.parametrize(
    "gene, variants, expected",
    [
        ("KRAS", {"activating"}, "activating"),
        ("KRAS", {"activating", "other"}, "activating"),
        ("KRAS", {"LoF"}, "LoF"),
        ("KRAS", {"LoF", "activating"}, "other"),  # ambiguous
        ("KRAS", {"LoF", "other"}, "other"),
        ("KRAS", {"other"}, "other"),
        ("TP53", {"other", "LoF"}, "LoF"),
        ("TP53", {"other"}, "other"),
    ],
)
def test_gene_status(gene, variants, expected):
    """
    Combination of the variants of one gene.

    :param expected: expected result
    :param gene: gene symbol
    :param variants: variant statuses of the gene
    """
    assert status_script.gene_status(variants, gene) == expected


def _all(value="wildtype", **changes):
    return {gene: value for gene in GENES} | changes


def test_aggregate_pathways():
    """Pathway rows are activating, other or wildtype."""
    assert status_script.aggregate_pathways(_all()) == {"PI3K_AKT": "wildtype", "RAS_MAPK": "wildtype"}
    result = status_script.aggregate_pathways(_all(PTEN="LoF", NF1="other"))
    assert result == {"PI3K_AKT": "activating", "RAS_MAPK": "other"}
    result = status_script.aggregate_pathways(_all(KRAS="activating", PIK3CA="other"))
    assert result == {"PI3K_AKT": "other", "RAS_MAPK": "activating"}
    assert status_script.aggregate_pathways(_all(NF1="LoF"))["RAS_MAPK"] == "activating"
    assert status_script.aggregate_pathways(_all(AKT1="activating"))["PI3K_AKT"] == "activating"


def _write_depmap(directory):
    pd.DataFrame(
        {"ModelID": ["ACH-1", "ACH-2", "ACH-3"], "RRID": ["CVCL_A", "CVCL_B", None]},
    ).to_csv(directory / "Model.csv", index=False)
    pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-1", "ACH-1", "ACH-2", "ACH-2"],
            "HugoSymbol": ["BRAF", "TP53", "KRAS", "KRAS", "OTHER"],
            "ProteinChange": ["p.V600E", "p.R175H", None, "p.G12D", "p.X1Y"],
            "Hotspot": [True, True, False, False, True],
            "LikelyLoF": [False, False, True, True, False],
            "OncogeneHighImpact": [True, False, False, False, False],
        }
    ).to_csv(directory / "OmicsSomaticMutations.csv", index=False)


def _cell_lines():
    return pd.DataFrame(
        {
            "cellosaurus_id": ["CVCL_B", "CVCL_A", "CVCL_C", "CVCL_A"],
            "cell_line_name": ["b", "a", "no-model", "a"],
        }
    )


def test_build_matrix(tmp_path):
    """
    Rows per cell line in input order, then the pathway blocks; cell lines without a model are left out.

    :param tmp_path: pytest temporary directory
    """
    _write_depmap(tmp_path)
    matrix = status_script.build_matrix(tmp_path, _cell_lines())
    assert len(matrix) == 2 * (len(GENES) + 2)
    assert matrix["sample"].tolist()[: len(GENES)] == ["b"] * len(GENES)
    assert matrix["gene"].tolist()[-4:] == ["RAS_MAPK", "RAS_MAPK", "PI3K_AKT", "PI3K_AKT"]
    status = matrix.set_index(["sample", "gene"])["status"]
    assert status["a", "BRAF"] == "activating"
    assert status["a", "TP53"] == "LoF"
    assert status["a", "KRAS"] == "other"  # LikelyLoF without protein change is not called truncating
    assert status["b", "KRAS"] == "LoF"  # LikelyLoF with protein change, no hotspot
    assert status["a", "EGFR"] == "wildtype"
    assert status["a", "RAS_MAPK"] == "activating"
    v600 = matrix.set_index(["sample", "gene"])["is_V600"]
    assert bool(v600["a", "BRAF"]) and not bool(v600["a", "TP53"])
    assert matrix.set_index(["sample", "gene"])["protein_changes"]["a", "BRAF"] == "p.V600E"


def test_download_depmap(tmp_path, monkeypatch):
    """
    A present file with the right md5 is skipped; a wrong download is an error.

    :param monkeypatch: pytest monkeypatch fixture
    :param tmp_path: pytest temporary directory
    """
    content = b"abc"
    md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(status_script, "DEPMAP_FILES", {"f.csv": ("https://example.org/f", md5)})
    (tmp_path / "f.csv").write_bytes(content)

    monkeypatch.setattr(status_script.urllib.request, "urlretrieve", lambda *args: pytest.fail("must not download"))
    status_script.download_depmap(tmp_path)

    (tmp_path / "f.csv").write_bytes(b"broken")
    monkeypatch.setattr(status_script.urllib.request, "urlretrieve", lambda url, path: open(path, "wb").write(content))
    status_script.download_depmap(tmp_path)
    assert (tmp_path / "f.csv").read_bytes() == content

    (tmp_path / "f.csv").unlink()
    monkeypatch.setattr(status_script.urllib.request, "urlretrieve", lambda url, path: open(path, "wb").write(b"x"))
    with pytest.raises(ValueError):
        status_script.download_depmap(tmp_path)


def test_main_defaults_use_all_datasets(tmp_path, monkeypatch):
    """
    Without arguments the matrix covers every dataset folder, including cell lines only in mutations.csv.

    :param monkeypatch: pytest monkeypatch fixture
    :param tmp_path: pytest temporary directory
    """
    depmap = tmp_path / "depmap_24Q4"
    depmap.mkdir()
    _write_depmap(depmap)
    (tmp_path / "D1").mkdir()
    (tmp_path / "D2").mkdir()
    _cell_lines().iloc[:2].to_csv(tmp_path / "D1" / "cell_line_names.csv", index=False)
    pd.DataFrame({"cellosaurus_id": ["CVCL_A"], "cell_line_name": ["a2"]}).to_csv(
        tmp_path / "D2" / "mutations.csv", index=False
    )
    monkeypatch.setattr(status_script, "download_depmap", lambda path: None)
    monkeypatch.setattr(sys, "argv", ["x", "--data_path", str(tmp_path)])
    status_script.main()
    matrix = pd.read_csv(tmp_path / "meta" / "target_mutation" / "functional_status_matrix.csv")
    assert set(matrix["cellosaurus_id"]) == {"CVCL_A", "CVCL_B"}

    monkeypatch.setattr(sys, "argv", ["x", "--data_path", str(tmp_path / "empty")])
    with pytest.raises(FileNotFoundError):
        status_script.main()


def test_curation_and_check_scripts_end_to_end(tmp_path, monkeypatch):
    """
    The curation script writes table and report; the check script reads them and changes nothing.

    :param monkeypatch: pytest monkeypatch fixture
    :param tmp_path: pytest temporary directory
    """
    directory = tmp_path / "meta" / "target_mutation"
    directory.mkdir(parents=True)
    cells = [f"CVCL_{i}" for i in range(40)]
    rows = [(c, g, "activating" if i < 10 and g == "BRAF" else "wildtype") for i, c in enumerate(cells) for g in GENES]
    pd.DataFrame(rows, columns=["cellosaurus_id", "gene", "status"]).to_csv(
        directory / "functional_status_matrix.csv", index=False
    )
    seed = pd.DataFrame(
        {
            "drug_name": ["a", "b"],
            "pubchem_id": ["1", "2"],
            "biomarker_gene": ["BRAF", "ALK"],
            "sensitive_status": ["activating", "activating"],
            "selectivity": ["selective", "selective"],
            "notes": ["", ""],
            "origin": ["existing", "draft"],
        }
    )
    seed.to_csv(tmp_path / "seed.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--data_path", str(tmp_path), "--seed", str(tmp_path / "seed.csv")])
    create_curated_drug_target_table.main()
    curated = pd.read_csv(directory / "curated_drug_target_variant.csv", dtype=str)
    assert curated["pubchem_id"].tolist() == ["1"]
    assert list(pd.read_csv(directory / "curated_drug_target_report.csv")["outcome"])[1].startswith("not representable")

    ds = tmp_path / "DS"
    ds.mkdir()
    auc = [0.3 + 0.01 * (i % 5) if i < 10 else 0.8 + 0.01 * (i % 5) for i in range(40)]
    pd.DataFrame({"cellosaurus_id": cells, "pubchem_id": "1", "AUC": auc}).to_csv(ds / "DS.csv", index=False)
    before = (directory / "curated_drug_target_variant.csv").read_bytes()
    monkeypatch.setattr(sys, "argv", ["x", "--data_path", str(tmp_path), "--datasets", "DS"])
    check_curated_drug_target_table.main()
    check = pd.read_csv(directory / "curated_drug_target_check.csv")
    assert check["summary"].tolist() == ["supported"]
    assert (directory / "curated_drug_target_variant.csv").read_bytes() == before

"""Builds the functional mutation status matrix (functional_status_matrix.csv) from raw DepMap 24Q4 files."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

#: Raw DepMap 24Q4 files (figshare article 27993248): name -> (download url, md5).
DEPMAP_FILES = {
    "Model.csv": ("https://ndownloader.figshare.com/files/51065297", "675210d17675f3517b0ce39a3c274f16"),
    "OmicsSomaticMutations.csv": (
        "https://ndownloader.figshare.com/files/51065732",
        "7bdba347a1602fe96d5654a74d6e52f1",
    ),
}
#: Genes of the matrix, in output order.
GENES = [
    "BRAF", "EGFR", "ALK", "KRAS", "NRAS", "PIK3CA", "MET", "KIT", "PDGFRA", "FLT3", "ABL1", "JAK2",
    "ERBB2", "RET", "ROS1", "FGFR1", "FGFR2", "FGFR3", "MAP2K1", "AKT1", "PTEN", "NF1", "TP53", "MDM2",
]  # fmt: skip
#: Tumour suppressors: a loss-of-function or hotspot variant means LoF. All other genes are treated as oncogenes.
TUMOUR_SUPPRESSORS = {"PTEN", "NF1", "TP53"}
#: Status written for a gene without any variant.
WILDTYPE = "wildtype"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_depmap(depmap_dir: Path) -> None:
    """
    Downloads the DepMap files that are missing or have a wrong checksum and verifies the md5 of every file.

    :param depmap_dir: target directory
    :raises ValueError: if a downloaded file does not match its md5
    """
    depmap_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, md5) in DEPMAP_FILES.items():
        path = depmap_dir / name
        if path.exists() and _md5(path) == md5:
            print(f"{name}: present, md5 ok, skipped")
            continue
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, path)  # noqa: S310 (fixed https figshare URLs)
        if _md5(path) != md5:
            raise ValueError(f"{name}: md5 mismatch after download (expected {md5}).")


def variant_status(
    gene: str, hotspot: bool, likely_lof: bool, oncogene_high_impact: bool, has_protein_change: bool
) -> str:
    """
    Classifies one variant.

    A truncating variant (LikelyLoF with a protein change) is LoF. A splice variant without protein change keeps the
    oncogene flags and can therefore be activating, as in the shipped table.

    :param gene: gene symbol
    :param hotspot: DepMap Hotspot flag
    :param likely_lof: DepMap LikelyLoF flag
    :param oncogene_high_impact: DepMap OncogeneHighImpact flag
    :param has_protein_change: whether DepMap reports a protein change
    :returns: activating, LoF or other
    """
    if gene in TUMOUR_SUPPRESSORS:
        return "LoF" if (likely_lof or hotspot) else "other"
    if likely_lof and has_protein_change and not hotspot:
        return "LoF"
    if hotspot or oncogene_high_impact:
        return "activating"
    return "other"


def gene_status(variant_statuses: set[str], gene: str) -> str:
    """
    Combines the variant statuses of one gene in one cell line.

    :param variant_statuses: statuses of all variants of the gene
    :param gene: gene symbol
    :returns: activating, LoF or other
    """
    if gene in TUMOUR_SUPPRESSORS:
        return "LoF" if "LoF" in variant_statuses else "other"
    if "LoF" in variant_statuses:
        return "LoF" if variant_statuses == {"LoF"} else "other"
    return "activating" if "activating" in variant_statuses else "other"


def aggregate_pathways(status: dict[str, str]) -> dict[str, str]:
    """
    Derives the two pathway pseudo-genes from the gene statuses of one cell line.

    :param status: gene -> status
    :returns: pathway -> status
    """

    def pathway(trigger: bool, components: list[str]) -> str:
        if trigger:
            return "activating"
        return "other" if any(status[g] != WILDTYPE for g in components) else WILDTYPE

    pi3k = ["PIK3CA", "PTEN", "AKT1"]
    ras = ["BRAF", "KRAS", "NRAS", "NF1"]
    return {
        "PI3K_AKT": pathway(
            status["PIK3CA"] == "activating" or status["PTEN"] == "LoF" or status["AKT1"] == "activating", pi3k
        ),
        "RAS_MAPK": pathway(
            status["BRAF"] == "activating"
            or status["KRAS"] == "activating"
            or status["NRAS"] == "activating"
            or status["NF1"] == "LoF",
            ras,
        ),
    }


def build_matrix(depmap_dir: Path, cell_lines: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the long-format status matrix.

    :param depmap_dir: directory with Model.csv and OmicsSomaticMutations.csv
    :param cell_lines: table with the columns cellosaurus_id and cell_line_name; defines which cell lines (and in
        which order) are in the matrix. Cell lines without a DepMap model are left out.
    :returns: table with sample, cellosaurus_id, gene, status, protein_changes, is_V600
    """
    models = pd.read_csv(depmap_dir / "Model.csv", usecols=["ModelID", "RRID"]).dropna(subset=["RRID"])
    models = models.drop_duplicates("RRID")
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=["ModelID", "HugoSymbol", "ProteinChange", "Hotspot", "LikelyLoF", "OncogeneHighImpact"],
        low_memory=False,
    )
    mutations = mutations[mutations["HugoSymbol"].isin(GENES)]
    mutations = mutations.assign(
        status=[
            variant_status(g, bool(h), bool(lof), bool(onc), bool(pc))
            for g, h, lof, onc, pc in zip(
                mutations["HugoSymbol"],
                mutations["Hotspot"],
                mutations["LikelyLoF"],
                mutations["OncogeneHighImpact"],
                mutations["ProteinChange"].notna(),
            )
        ]
    )
    variants = {key: frame for key, frame in mutations.groupby(["ModelID", "HugoSymbol"])}
    cell_lines = cell_lines.drop_duplicates("cellosaurus_id").merge(
        models, left_on="cellosaurus_id", right_on="RRID", how="inner"
    )
    rows = []
    pathway_rows: dict[str, list] = {"RAS_MAPK": [], "PI3K_AKT": []}
    for sample, cellosaurus_id, model_id in zip(
        cell_lines["cell_line_name"], cell_lines["cellosaurus_id"], cell_lines["ModelID"]
    ):
        status: dict[str, str] = {}
        records = []
        for gene in GENES:
            frame = variants.get((model_id, gene))
            if frame is None:
                status[gene] = WILDTYPE
                changes: float | str = np.nan
                v600 = False
            else:
                status[gene] = gene_status(set(frame["status"]), gene)
                protein = sorted(frame["ProteinChange"].dropna().astype(str).unique())
                changes = ";".join(protein) if len(protein) else np.nan
                v600 = bool(frame["ProteinChange"].astype(str).str.contains("V600").any())
            records.append((gene, status[gene], changes, v600))
        for pathway_name, pathway_status in aggregate_pathways(status).items():
            pathway_rows[pathway_name].append((sample, cellosaurus_id, pathway_name, pathway_status, np.nan, False))
        rows += [(sample, cellosaurus_id, *record) for record in records]
    # as in the shipped table: gene rows per cell line first, then one block per pathway sorted by cell line name
    for name in ("RAS_MAPK", "PI3K_AKT"):
        rows += sorted(pathway_rows[name], key=lambda row: row[0])
    return pd.DataFrame(rows, columns=["sample", "cellosaurus_id", "gene", "status", "protein_changes", "is_V600"])


def main():
    """
    Download the DepMap files and write functional_status_matrix.csv.

    :raises FileNotFoundError: if no cell_line_names.csv is found below the data path
    """
    parser = argparse.ArgumentParser(description="Build the functional mutation status matrix from DepMap 24Q4.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Data directory. Without --cell_line_names, all <data_path>/*/cell_line_names.csv define the cell lines.",
    )
    parser.add_argument(
        "--cell_line_names",
        type=str,
        nargs="+",
        default=None,
        help="cell_line_names.csv file(s) (cellosaurus_id, cell_line_name) defining the cell lines of the matrix.",
    )
    parser.add_argument(
        "--depmap_dir",
        type=str,
        default=None,
        help="Where the raw files are stored (default: <data_path>/depmap_24Q4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: <data_path>/meta/target_mutation/functional_status_matrix.csv).",
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    depmap_dir = Path(args.depmap_dir) if args.depmap_dir else data_path / "depmap_24Q4"
    name_files = args.cell_line_names or sorted(str(path) for path in data_path.glob("*/cell_line_names.csv"))
    if not name_files:
        raise FileNotFoundError(f"No cell_line_names.csv found below {data_path}.")
    # also the cell lines of the binary mutation tables: some are in no cell_line_names.csv
    mutation_files = [] if args.cell_line_names else sorted(str(path) for path in data_path.glob("*/mutations.csv"))
    download_depmap(depmap_dir)
    id_columns = ["cellosaurus_id", "cell_line_name"]
    cell_lines = pd.concat([pd.read_csv(path, dtype=str, usecols=id_columns) for path in name_files + mutation_files])
    matrix = build_matrix(depmap_dir, cell_lines[id_columns])
    output = Path(args.output or data_path / "meta" / "target_mutation" / "functional_status_matrix.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, index=False)
    print(f"{output}: {matrix['cellosaurus_id'].nunique()} cell lines, {len(matrix)} rows")


if __name__ == "__main__":
    main()

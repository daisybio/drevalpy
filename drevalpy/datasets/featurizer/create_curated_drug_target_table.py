"""Builds curated_drug_target_variant.csv from the seed table. Needs no response data."""

import argparse
from pathlib import Path

import pandas as pd

from drevalpy.datasets.featurizer import create_target_mutation_seed

#: Drivers that are fusions or amplifications. The status matrix only knows point mutations, so a drug that acts on
#: them cannot be represented.
NOT_REPRESENTABLE = {"ALK", "MET", "ERBB2", "FGFR1", "FGFR4"}
STATUSES = {"activating", "LoF", "other", "wildtype"}
OUTPUT_COLUMNS = ["drug_name", "biomarker_gene", "sensitive_status", "evidence_type", "evidence_level", "pubchem_id"]
SEED_NAME = "target_mutation_seed.csv"
#: CIViC evidence levels from best to worst; a rule without a level (literature) ranks last.
LEVEL_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def curate(seed: pd.DataFrame, genes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keeps exactly one seed row per drug.

    The seed can hold several biomarker genes per drug, the model uses one. Only rows whose gene the status matrix can
    represent are candidates. Among them the row with the best CIViC evidence level wins, then the one with the most
    evidence items, then the first in the seed. The choice depends only on the mechanism of the drug (the seed) and on
    the columns of the status matrix, never on response data, so the table is the same for every dataset.

    :param seed: seed table as written by create_target_mutation_seed.py
    :param genes: genes of the status matrix
    :returns: the curated table with one row per drug and a report with one row per seed entry
    :raises ValueError: if the seed has an unknown sensitive status
    """
    if not set(seed["sensitive_status"]) <= STATUSES:
        raise ValueError(f"Unknown sensitive_status in the seed: {set(seed['sensitive_status']) - STATUSES}")
    outcome = pd.Series("kept", index=seed.index)
    outcome[seed["biomarker_gene"].isin(NOT_REPRESENTABLE)] = "not representable (fusion or amplification)"
    outcome[~seed["biomarker_gene"].isin(genes | NOT_REPRESENTABLE)] = "gene not in the status matrix"
    candidates = seed[outcome == "kept"]
    order = candidates.assign(
        _level=candidates["evidence_level"].map(LEVEL_RANK).fillna(len(LEVEL_RANK)),
        _items=-pd.to_numeric(candidates["n_evidence_items"]).fillna(0),
    ).sort_values(["_level", "_items"], kind="stable")
    chosen = order.drop_duplicates("pubchem_id").index
    outcome[candidates.index.difference(chosen)] = "other row of the drug selected"
    report = seed.assign(outcome=outcome)
    return seed.loc[chosen.sort_values(), OUTPUT_COLUMNS].reset_index(drop=True), report


def main():
    """Read the seed (built first if missing) and the status matrix genes, write the curated table and a report."""
    parser = argparse.ArgumentParser(description="Build the curated drug target table.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with meta/target_mutation.")
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help=f"Seed table (default: <output>/{SEED_NAME}). If the file does not exist it is built with "
        "create_target_mutation_seed.py (needs internet).",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Directory (default: <data_path>/meta/target_mutation)."
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.data_path) / "meta" / "target_mutation"
    seed_file = Path(args.seed) if args.seed else output / SEED_NAME
    if not seed_file.exists():
        print(f"{seed_file} does not exist, building it with create_target_mutation_seed.py")
        create_target_mutation_seed.main(["--data_path", args.data_path, "--out", str(seed_file)])
    genes = set(pd.read_csv(output / "functional_status_matrix.csv", usecols=["gene"], dtype=str)["gene"])
    curated, report = curate(pd.read_csv(seed_file, dtype=str, keep_default_na=False), genes)
    curated.to_csv(output / "curated_drug_target_variant.csv", index=False)
    report.to_csv(output / "curated_drug_target_report.csv", index=False)
    print(report["outcome"].value_counts().to_string())
    print(f"{output}: {len(curated)} drugs")


if __name__ == "__main__":
    main()

"""Builds curated_drug_target_variant.csv from the hand-kept seed table. Needs no response data."""

import argparse
from pathlib import Path

import pandas as pd

#: Drivers that are fusions or amplifications. The status matrix only knows point mutations, so a drug that acts on
#: them cannot be represented.
NOT_REPRESENTABLE = {"ALK", "MET", "ERBB2", "FGFR1", "FGFR4"}
STATUSES = {"activating", "LoF", "other", "wildtype"}
OUTPUT_COLUMNS = ["drug_name", "biomarker_gene", "sensitive_status", "selectivity", "notes", "pubchem_id"]
SEED_FILE = Path(__file__).with_name("target_mutation_seed.csv")


def curate(seed: pd.DataFrame, genes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keeps the seed entries whose gene the status matrix can represent.

    The choice depends only on the mechanism of the drug (the seed) and on the columns of the status matrix, never on
    response data, so the table is the same for every dataset.

    :param seed: seed table with the columns of the curated table and ``origin``
    :param genes: genes of the status matrix
    :returns: the curated table and a report with one row per seed entry
    :raises ValueError: if the seed has an unknown sensitive status or a duplicate pubchem_id
    """
    if not set(seed["sensitive_status"]) <= STATUSES:
        raise ValueError(f"Unknown sensitive_status in the seed: {set(seed['sensitive_status']) - STATUSES}")
    if seed["pubchem_id"].duplicated().any():
        raise ValueError(f"Duplicate pubchem_id in the seed: {seed['pubchem_id'][seed['pubchem_id'].duplicated()]}")
    outcome = pd.Series("kept", index=seed.index)
    outcome[seed["biomarker_gene"].isin(NOT_REPRESENTABLE)] = "not representable (fusion or amplification)"
    outcome[~seed["biomarker_gene"].isin(genes | NOT_REPRESENTABLE)] = "gene not in the status matrix"
    report = seed.assign(outcome=outcome)
    return seed[outcome == "kept"][OUTPUT_COLUMNS].reset_index(drop=True), report


def main():
    """Read the seed and the gene list of the status matrix, write curated_drug_target_variant.csv and a report."""
    parser = argparse.ArgumentParser(description="Build the curated drug target table.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with meta/target_mutation.")
    parser.add_argument("--seed", type=str, default=str(SEED_FILE), help="Hand-kept seed table.")
    parser.add_argument(
        "--output", type=str, default=None, help="Directory (default: <data_path>/meta/target_mutation)."
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.data_path) / "meta" / "target_mutation"
    genes = set(pd.read_csv(output / "functional_status_matrix.csv", usecols=["gene"], dtype=str)["gene"])
    curated, report = curate(pd.read_csv(args.seed, dtype=str), genes)
    curated.to_csv(output / "curated_drug_target_variant.csv", index=False)
    report.to_csv(output / "curated_drug_target_report.csv", index=False)
    print(report["outcome"].value_counts().to_string())
    print(f"{output}: {len(curated)} drugs")


if __name__ == "__main__":
    main()

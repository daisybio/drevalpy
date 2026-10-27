"""Diagnostic: checks the entries of the curated drug target table against the response data. Changes nothing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

#: Thresholds of the scale-free check (Cohen d of the sensitive group against the rest, Mann-Whitney p).
MIN_PER_GROUP = 5
MAX_D = -0.3
MAX_P = 0.01
#: A group this small cannot refute an entry.
MIN_TO_REFUTE = 10


def check_candidate(
    response: pd.DataFrame, status: pd.DataFrame, pubchem_id: str, gene: str, sensitive_status: str
) -> dict:
    """
    Compares the AUC of the cell lines with the sensitive status against all others for one drug.

    :param response: response data with the columns cellosaurus_id, pubchem_id and AUC
    :param status: status matrix, cellosaurus_id x gene
    :param pubchem_id: the drug
    :param gene: biomarker gene
    :param sensitive_status: status that should make cells sensitive
    :returns: n, d, p and the verdict (supported, underpowered, no signal, untestable or no data)
    """
    rows = response[(response["pubchem_id"] == pubchem_id) & response["cellosaurus_id"].isin(status.index)]
    if rows.empty:
        return {"n": 0, "d": np.nan, "p": np.nan, "verdict": "no data"}
    cell_status = status.loc[rows["cellosaurus_id"], gene].to_numpy()
    sensitive, rest = (
        rows["AUC"].to_numpy()[cell_status == sensitive_status],
        rows["AUC"].to_numpy()[cell_status != sensitive_status],
    )
    if len(sensitive) < MIN_PER_GROUP or len(rest) < MIN_PER_GROUP:
        return {"n": len(sensitive), "d": np.nan, "p": np.nan, "verdict": "untestable"}
    pooled = np.sqrt(
        ((len(sensitive) - 1) * sensitive.var(ddof=1) + (len(rest) - 1) * rest.var(ddof=1))
        / (len(sensitive) + len(rest) - 2)
    )
    d = (sensitive.mean() - rest.mean()) / pooled if pooled > 0 else 0.0
    p = mannwhitneyu(sensitive, rest).pvalue
    if d <= MAX_D and p < MAX_P:
        verdict = "supported"
    elif d <= MAX_D or len(sensitive) < MIN_TO_REFUTE:
        verdict = "underpowered"
    elif d < 0 and p < MAX_P:
        verdict = "weak"
    else:
        verdict = "no signal"
    return {"n": len(sensitive), "d": d, "p": p, "verdict": verdict}


def summarize(tests: dict[str, dict]) -> str:
    """
    Combines the results of the datasets for one candidate.

    :param tests: dataset -> result of check_candidate
    :returns: supported (one dataset supports it), underpowered (a large effect without significance, or too few cell
        lines), weak (significant but small effect in the right direction), refuted (a testable dataset shows nothing
        and none shows more) or untestable
    """
    verdicts = {t["verdict"] for t in tests.values()}
    if "supported" in verdicts:
        return "supported"
    if "underpowered" in verdicts:
        return "underpowered"
    if "weak" in verdicts:
        return "weak"
    if "no signal" in verdicts:
        return "refuted"
    return "untestable"


def main():
    """Write curated_drug_target_check.csv with the verdict per drug and dataset. The curated table stays as it is."""
    parser = argparse.ArgumentParser(description="Check the curated drug target table against response data.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with the dataset folders.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["CTRPv2", "GDSC1", "CCLE"], help="Response data.")
    parser.add_argument(
        "--target_mutation_path", type=str, default=None, help="Default: <data_path>/meta/target_mutation"
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    directory = Path(args.target_mutation_path) if args.target_mutation_path else data_path / "meta" / "target_mutation"
    matrix = pd.read_csv(directory / "functional_status_matrix.csv", dtype=str)
    status = matrix.pivot(index="cellosaurus_id", columns="gene", values="status")
    curated = pd.read_csv(directory / "curated_drug_target_variant.csv", dtype=str)
    responses = {
        name: pd.read_csv(
            data_path / name / f"{name}.csv", dtype={"pubchem_id": str}, usecols=["cellosaurus_id", "pubchem_id", "AUC"]
        ).dropna()
        for name in args.datasets
    }
    rows = []
    for entry in curated.itertuples(index=False):
        tests = {
            name: check_candidate(data, status, entry.pubchem_id, entry.biomarker_gene, entry.sensitive_status)
            for name, data in responses.items()
        }
        rows.append(
            {"drug_name": entry.drug_name, "biomarker_gene": entry.biomarker_gene, "summary": summarize(tests)}
            | {f"{name}_{key}": value for name, test in tests.items() for key, value in test.items()}
        )
    report = pd.DataFrame(rows)
    report.to_csv(directory / "curated_drug_target_check.csv", index=False)
    print(report["summary"].value_counts().to_string())


if __name__ == "__main__":
    main()

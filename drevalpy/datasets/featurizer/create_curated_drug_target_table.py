"""Builds curated_drug_target_variant.csv from a hand-kept seed table and a check against the response data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

#: Drivers that are fusions or amplifications. The status matrix only knows point mutations, so a drug that acts on
#: them cannot be represented, and more cell lines do not help.
NOT_REPRESENTABLE = {"ALK", "MET", "ERBB2", "FGFR1", "FGFR4"}
#: A pathway label is not a mutable gene. Its drugs are tested on the candidates below, in this order.
PSEUDO_GENES = {"PI3K_AKT": [("PIK3CA", "activating"), ("PTEN", "LoF")]}
#: Thresholds of the scale-free check (Cohen d of the sensitive group against the rest, Mann-Whitney p).
MIN_PER_GROUP = 5
MAX_D = -0.3
MAX_P = 0.01
#: A group this small cannot refute an entry.
MIN_TO_REFUTE = 10
OUTPUT_COLUMNS = ["drug_name", "biomarker_gene", "sensitive_status", "selectivity", "notes", "pubchem_id", "evidence"]
SEED_FILE = Path(__file__).with_name("target_mutation_seed.csv")


def test_candidate(
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

    :param tests: dataset -> result of test_candidate
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


def curate(
    seed: pd.DataFrame, responses: dict[str, pd.DataFrame], status: pd.DataFrame, draft_policy: str = "supported"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the rules to the seed table.

    Existing entries stay unless they are not representable or refuted. New entries (drafts) enter only if they are
    supported (``draft_policy`` supported) or at least not refuted (not_refuted). The data may therefore only demote an
    existing entry, but with the policy supported it also admits a draft.

    :param seed: seed table with the columns of the curated table and ``origin`` (existing or draft)
    :param responses: dataset -> response data
    :param status: status matrix, cellosaurus_id x gene
    :param draft_policy: supported or not_refuted
    :returns: the curated table and a report with one row per seed entry
    """
    kept, report = [], []
    for entry in seed.itertuples(index=False):
        candidates = PSEUDO_GENES.get(entry.biomarker_gene, [(entry.biomarker_gene, entry.sensitive_status)])
        chosen, outcome, evidence = None, "", ""
        if entry.biomarker_gene in NOT_REPRESENTABLE:
            outcome = "not representable"
        else:
            results = []
            for gene, sensitive_status in candidates:
                tests = {
                    name: test_candidate(data, status, entry.pubchem_id, gene, sensitive_status)
                    for name, data in responses.items()
                }
                results.append((gene, sensitive_status, tests, summarize(tests)))
            usable = [r for r in results if r[3] != "refuted"]
            gene, sensitive_status, tests, summary = usable[0] if usable else results[0]
            if not usable:
                outcome = "refuted"
            elif entry.origin == "draft" and not (summary == "supported" or draft_policy == "not_refuted"):
                outcome = f"held ({summary})"
            else:
                outcome, chosen = "kept" if entry.origin == "existing" else "added", (gene, sensitive_status)
            evidence = "; ".join(
                f"{name}: n={t['n']} d={t['d']:.2f} p={t['p']:.1g}"
                for name, t in tests.items()
                if t["verdict"] not in ("no data", "untestable")
            )
            evidence = f"{summary}" + (f" ({evidence})" if evidence else "")
        report.append({**entry._asdict(), "outcome": outcome, "evidence": evidence})
        if chosen is not None:
            note = entry.notes + (
                f" [{entry.biomarker_gene} -> {chosen[0]}]" if chosen[0] != entry.biomarker_gene else ""
            )
            kept.append({**entry._asdict(), "biomarker_gene": chosen[0], "sensitive_status": chosen[1], "notes": note,
                         "evidence": evidence})  # fmt: skip
    return pd.DataFrame(kept)[OUTPUT_COLUMNS], pd.DataFrame(report)


def main():
    """Read the seed and the response data, write curated_drug_target_variant.csv and a report."""
    parser = argparse.ArgumentParser(description="Build the curated drug target table.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with the dataset folders.")
    parser.add_argument("--seed", type=str, default=str(SEED_FILE), help="Hand-kept seed table.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["CTRPv2", "GDSC1", "CCLE"], help="Response data.")
    parser.add_argument("--matrix", type=str, default=None, help="functional_status_matrix.csv (default: in --output).")
    parser.add_argument("--draft_policy", type=str, default="supported", choices=["supported", "not_refuted"])
    parser.add_argument(
        "--output", type=str, default=None, help="Directory (default: <data_path>/meta/target_mutation)."
    )
    args = parser.parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output) if args.output else data_path / "meta" / "target_mutation"
    matrix = pd.read_csv(args.matrix or output / "functional_status_matrix.csv", dtype=str)
    status = matrix.pivot(index="cellosaurus_id", columns="gene", values="status")
    responses = {
        name: pd.read_csv(
            data_path / name / f"{name}.csv", dtype={"pubchem_id": str}, usecols=["cellosaurus_id", "pubchem_id", "AUC"]
        ).dropna()
        for name in args.datasets
    }
    seed = pd.read_csv(args.seed, dtype=str)
    curated, report = curate(seed, responses, status, args.draft_policy)
    output.mkdir(parents=True, exist_ok=True)
    curated.to_csv(output / "curated_drug_target_variant.csv", index=False)
    report.to_csv(output / "curated_drug_target_report.csv", index=False)
    print(report["outcome"].value_counts().to_string())
    print(f"{output}: {len(curated)} drugs")


if __name__ == "__main__":
    main()

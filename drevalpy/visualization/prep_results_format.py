"""Helpers to enrich raw evaluation tables before plotting."""

from __future__ import annotations

import pandas as pd
from upath import UPath as Path

from ..data.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from .normalize_metrics import normalize_metrics_by_mean_effects


def load_drug_and_cell_line_metadata(path_data: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Walk ``path_data`` and collect drug and cell-line name mappings.

    :param path_data: Root directory to search for ``drug_names.csv`` and

    :returns: Tuple of ``pubchem_id → drug_name`` and cell-line id → cellosaurus id
    :returns: mappings.
    """
    drug_metadata: dict[str, str] = {}
    cell_line_metadata: dict[str, str] = {}
    for file in Path(path_data).rglob("*.csv"):
        if file.name == "drug_names.csv":
            drug_names = pd.read_csv(file)
            drug_names["pubchem_id"] = drug_names["pubchem_id"].astype(str)
            drug_metadata.update(zip(drug_names["pubchem_id"], drug_names["drug_name"], strict=False))
        elif file.name == "cell_line_names.csv":
            cell_line_metadata.update(_cell_line_name_mapping(file))
    return drug_metadata, cell_line_metadata


def _cell_line_name_mapping(path: Path) -> dict[str, str]:
    cell_line_names = pd.read_csv(path)
    try:
        cellosaurus_ids = cell_line_names["cellosaurus_id"].astype(str)
        n_missing = cellosaurus_ids.isna().sum()
        fill_values = [f"unknown_id_{i}" for i in range(n_missing)]
        cellosaurus_ids = cellosaurus_ids.where(
            cellosaurus_ids.notna(),
            pd.Series(fill_values, index=cellosaurus_ids[cellosaurus_ids.isna()].index),
        )
    except KeyError:
        cellosaurus_ids = pd.Series([f"unknown_id_{i}" for i in range(len(cell_line_names))])
    return dict(zip(cell_line_names[CELL_LINE_IDENTIFIER], cellosaurus_ids, strict=False))


def add_index_columns_from_model(eval_results: pd.DataFrame) -> pd.DataFrame:
    """Split the model index into algorithm, setting, test mode, and CV split columns.

    :param eval_results: Evaluation results indexed by encoded model run names.

    :returns: Input table with parsed index columns prepended.
    """
    new_columns = eval_results.index.str.split("_", expand=True).to_frame()
    new_columns.columns = ["algorithm", "rand_setting", "test_mode", "split", "CV_split"]
    new_columns.index = eval_results.index
    return pd.concat([new_columns.drop("split", axis=1), eval_results], axis=1)


def enrich_eval_results_per_drug(
    eval_results_per_drug: pd.DataFrame | None,
    drug_metadata: dict[str, str],
) -> pd.DataFrame | None:
    """Add drug names and parsed model fields to per-drug results.

    :param eval_results_per_drug: Per-drug evaluation table, or ``None``.
    :param drug_metadata: Mapping from drug id to human-readable name.

    :returns: Enriched per-drug table, or ``None`` if input was ``None``.
    """
    if eval_results_per_drug is None:
        return None
    eval_results_per_drug = eval_results_per_drug.copy()
    eval_results_per_drug[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = eval_results_per_drug[
        "model"
    ].str.split("_", expand=True)
    eval_results_per_drug["drug_name"] = [drug_metadata[drug] for drug in eval_results_per_drug["drug"]]
    return eval_results_per_drug.rename(columns={"drug": DRUG_IDENTIFIER})


def enrich_eval_results_per_cell_line(
    eval_results_per_cell_line: pd.DataFrame | None,
    cell_line_metadata: dict[str, str],
) -> pd.DataFrame | None:
    """Add cellosaurus ids and parsed model fields to per-cell-line results.

    :param eval_results_per_cell_line: Per-cell-line evaluation table, or ``None``.
    :param cell_line_metadata: Mapping from cell-line id to cellosaurus id.

    :returns: Enriched per-cell-line table, or ``None`` if input was ``None``.
    """
    if eval_results_per_cell_line is None:
        return None
    eval_results_per_cell_line = eval_results_per_cell_line.copy()
    eval_results_per_cell_line[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = (
        eval_results_per_cell_line["model"].str.split("_", expand=True)
    )
    eval_results_per_cell_line["cellosaurus_id"] = [
        cell_line_metadata[cell_line] for cell_line in eval_results_per_cell_line["cell_line"]
    ]
    return eval_results_per_cell_line.rename(columns={"cell_line": CELL_LINE_IDENTIFIER})


def enrich_true_vs_pred(
    t_vs_p: pd.DataFrame,
    drug_metadata: dict[str, str],
    cell_line_metadata: dict[str, str],
) -> pd.DataFrame:
    """Add metadata columns and parsed model fields to true-versus-predicted rows.

    :param t_vs_p: True versus predicted values table.
    :param drug_metadata: Mapping from drug id to human-readable name.
    :param cell_line_metadata: Mapping from cell-line id to cellosaurus id.

    :returns: Enriched true-versus-predicted table with identifier columns.
    """
    t_vs_p = t_vs_p.copy()
    t_vs_p[["algorithm", "rand_setting", "test_mode", "split", "CV_split"]] = t_vs_p["model"].str.split(
        "_", expand=True
    )
    t_vs_p = t_vs_p.drop("split", axis=1)
    t_vs_p["drug_name"] = [drug_metadata[drug] for drug in t_vs_p["drug"]]
    t_vs_p["cellosaurus_id"] = [cell_line_metadata[cell_line] for cell_line in t_vs_p["cell_line"]]
    t_vs_p = t_vs_p.rename(columns={"cell_line": CELL_LINE_IDENTIFIER, "drug": DRUG_IDENTIFIER})
    t_vs_p[DRUG_IDENTIFIER] = t_vs_p[DRUG_IDENTIFIER].astype(str)
    return t_vs_p


def apply_mean_effects_normalization(
    eval_results: pd.DataFrame,
    t_vs_p: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize metrics using NaiveMeanEffectsPredictor baselines.

    :param eval_results: Overall evaluation results.
    :param t_vs_p: True versus predicted values for all models.

    :returns: ``eval_results`` merged with normalized metric columns.

    :raises ValueError: If ``NaiveMeanEffectsPredictor`` is not in the results.
    """
    if "NaiveMeanEffectsPredictor" not in eval_results["algorithm"].unique():
        raise ValueError(
            "NaiveMeanEffectsPredictor not found in evaluation results. "
            "Please check if the evaluation was run correctly."
        )
    return normalize_metrics_by_mean_effects(evaluation_results=eval_results, true_vs_pred=t_vs_p)

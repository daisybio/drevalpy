"""Leaderboard parity case: the 1.6.1 leaderboard table against the v2 one, on identical predictions.

No model is trained. Both sides receive the same prediction table (algorithm, CV split, sample, drug,
``y_true``, ``y_pred``) and build the leaderboard from it:

* 1.6.1 evaluates every (algorithm, split) with ``evaluate``, normalizes the metrics by the
  ``NaiveMeanEffectsPredictor`` predictions (``visualization.utils._normalize_metrics_by_mean_effects``),
  writes ``evaluation_results.csv`` and ``true_vs_pred.csv`` as ``drevalpy report`` does, and reads them back
  with ``create_leaderboard.load_results`` (normalized PCC, RMSE, raw Pearson, per-drug Pearson);
* v2 builds ``RunResult`` objects, ``ExperimentResult.normalize``-s them and calls ``build_leaderboard_df``.

Variants:

``main``
    Five algorithms over six splits of eight drugs, all predictions varying within every drug. The whole
    table (order of the rows included) is a parity step of class ``exact``. The same data also feed the
    critical difference diagram: Friedman p-value, ranks and the Conover-Friedman post-hoc table with
    ``p_adjust="fdr_bh"`` (1.6.1 through ``CriticalDifferencePlot``, scikit-posthocs 0.17.1 in the 1.6.1
    venv; v2 through the call ``CriticalDifferenceVisualization`` makes, scikit-posthocs of the v2 venv).
``constant``
    Adds ``NaiveDrugMeanPredictor`` (constant within every drug) and a drug with a single pair per split.
    Everything but the per-drug Pearson is a parity step; the per-drug Pearson is a recorded deviation, see
    ``FINDINGS``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.v1_parity._bridge import Case
from tools.v1_parity.compare import DEVIATION, PARITY, StepSpec

NAME = "Leaderboard"

REFERENCE = "NaiveMeanEffectsPredictor"
N_SPLITS = 6
N_SAMPLES_PER_SPLIT = 10
VARIANTS: dict[str, dict[str, Any]] = {
    "main": {
        "algorithms": [REFERENCE, "ElasticNet", "RandomForest", "XGBoost", "SimpleNeuralNetwork"],
        "noise": [1.0, 0.9, 0.5, 0.45, 0.7],
        "n_drugs": 8,
        "single_pair_drug": False,
    },
    "constant": {
        "algorithms": [REFERENCE, "NaiveDrugMeanPredictor", "ElasticNet", "RandomForest"],
        "noise": [1.0, 0.0, 0.9, 0.5],
        "n_drugs": 6,
        "single_pair_drug": True,
    },
}
COLUMNS = ("PCC", "PCC_std", "RMSE", "RMSE_std", "Pearson", "Pearson_std", "Pearson_per_drug", "Pearson_per_drug_std")
CD_METRIC = "Pearson"

FINDINGS = [
    "1.6.1 keeps the NaiveMeanEffectsPredictor as a baseline row of the leaderboard and of the critical "
    "difference diagram (normalized against itself: PCC 0, RMSE and raw metrics of its own predictions). "
    "`ExperimentResult.normalize` of v2 used to drop it; it now returns array-free summaries of the reference runs "
    "in `reference_runs`, which the leaderboard table and the critical difference data frame list.",
    "1.6.1 reports the RMSE and the raw Pearson of the unnormalized predictions next to the normalized Pearson. "
    "v2 `normalize` replaces the run by its residuals, so the unnormalized pooled metrics and the per-drug Pearson "
    "are kept on the run (`RunResult.raw_metrics`, `raw_per_drug_pearson`).",
    "The per-drug Pearson of 1.6.1 averages `evaluation.pearson` over the drugs of a split. That function returns 0.0 "
    "for a constant prediction, and a drug with one pair has a constant prediction, so such drugs count as 0.0. v2 "
    "computes it with `_grouped_evaluation`, the shared rule of the per-drug comparison plots: a drug whose "
    "predictions or response are constant, or with fewer than two pairs, is undefined and skipped, so a baseline "
    "that predicts a constant per drug has no per-drug Pearson (the panel leaves it out) where 1.6.1 draws 0.000.",
]


def _variant_arrays(seed: int, setting: dict[str, Any]) -> dict[str, np.ndarray]:
    """Draw one prediction table.

    :param seed: Seed of the generator.
    :param setting: Entry of :data:`VARIANTS`.
    :returns: ``fold``, ``sample``, ``drug``, ``y_true`` per row and ``y_pred`` of shape (algorithms, rows).
    """
    rng = np.random.default_rng(seed)
    n_drugs = setting["n_drugs"]
    drug_effect = rng.normal(scale=2.0, size=n_drugs + 1)
    folds, samples, drugs, truth, effects = [], [], [], [], []
    for fold in range(N_SPLITS):
        sample_effect = rng.normal(size=N_SAMPLES_PER_SPLIT)
        for s in range(N_SAMPLES_PER_SPLIT):
            for d in range(n_drugs):
                folds.append(fold)
                samples.append(f"S{fold}_{s:02d}")
                drugs.append(f"D{d:02d}")
                effects.append((drug_effect[d], sample_effect[s]))
        if setting["single_pair_drug"]:
            folds.append(fold)
            samples.append(f"S{fold}_00")
            drugs.append("D_single")
            effects.append((drug_effect[n_drugs], sample_effect[0]))
    drug_part, sample_part = np.array(effects).T
    truth = drug_part + sample_part + rng.normal(scale=0.5, size=len(drug_part))
    order = rng.permutation(len(truth))
    drug_labels = np.array(drugs)
    predictions = []
    for algorithm, noise in zip(setting["algorithms"], setting["noise"], strict=True):
        if algorithm == REFERENCE:
            pred = drug_part + 0.6 * sample_part + rng.normal(scale=noise * 0.1, size=len(truth))
        elif algorithm == "NaiveDrugMeanPredictor":
            means = {d: truth[drug_labels == d].mean() for d in np.unique(drug_labels)}
            pred = np.array([means[d] for d in drug_labels])
        else:
            pred = truth + rng.normal(scale=noise, size=len(truth))
        predictions.append(pred)
    return {
        "fold": np.array(folds)[order].astype(np.int64),
        "sample": np.array(samples)[order],
        "drug": drug_labels[order],
        "y_true": truth[order],
        "y_pred": np.array(predictions)[:, order],
    }


def build_case(seed: int) -> Case:
    """Draw the prediction tables of every variant.

    :param seed: Seed of the input generator.
    :returns: The case; arrays are named ``<variant>/<column>``.
    """
    arrays: dict[str, np.ndarray] = {}
    for index, (variant, setting) in enumerate(VARIANTS.items()):
        for key, value in _variant_arrays(seed + index, setting).items():
            arrays[f"{variant}/{key}"] = value
    return Case(model=NAME, seed=seed, arrays=arrays, meta={"variants": VARIANTS})


def steps() -> list[StepSpec]:
    """Declare the compared outputs.

    :returns: Exact parity steps for the whole table and the critical difference statistics of ``main``, and
        for everything of ``constant`` except the per-drug Pearson, which is a deviation.
    """
    specs = []
    for variant in VARIANTS:
        specs.append(StepSpec(name=f"{variant}/algorithm", description=f"{variant}: model order of the table"))
        specs.append(StepSpec(name=f"{variant}/is_baseline", description=f"{variant}: baseline flags"))
        for column in COLUMNS:
            per_drug = column.startswith("Pearson_per_drug") and variant == "constant"
            specs.append(
                StepSpec(
                    name=f"{variant}/{column}",
                    scope=DEVIATION if per_drug else PARITY,
                    description=f"{variant}: {column} of every model",
                    explanation=FINDINGS[2] if per_drug else "",
                )
            )
    specs += [
        StepSpec(name="main/cd/friedman_p", description="Friedman p-value over the splits (scipy)"),
        StepSpec(name="main/cd/ranks", description="average ranks of the normalized Pearson over the splits"),
        StepSpec(
            name="main/cd/conover",
            description="Conover-Friedman post-hoc p-values, p_adjust=fdr_bh (scikit-posthocs)",
        ),
    ]
    return specs


def _tables(case: Case, variant: str):
    """Rebuild the long prediction table of one variant.

    :param case: The shared case.
    :param variant: Key of :data:`VARIANTS`.
    :returns: A pandas frame with one row per (algorithm, pair).
    """
    import pandas as pd

    setting = VARIANTS[variant]
    frames = []
    for index, algorithm in enumerate(setting["algorithms"]):
        frames.append(
            pd.DataFrame(
                {
                    "algorithm": algorithm,
                    "CV_split": case.arrays[f"{variant}/fold"].astype(str),
                    "cell_line_name": case.arrays[f"{variant}/sample"],
                    "drug_name": case.arrays[f"{variant}/drug"],
                    "y_true": case.arrays[f"{variant}/y_true"],
                    "y_pred": case.arrays[f"{variant}/y_pred"][index],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _table_arrays(variant: str, table) -> dict[str, np.ndarray]:
    """Convert a leaderboard table to the output arrays.

    :param variant: Key of :data:`VARIANTS`.
    :param table: Frame with ``algorithm``, :data:`COLUMNS` and ``is_baseline``.
    :returns: Arrays keyed ``<variant>/<column>``.
    """
    out = {f"{variant}/algorithm": table["algorithm"].to_numpy().astype(str)}
    out[f"{variant}/is_baseline"] = table["is_baseline"].to_numpy().astype(np.int64)
    for column in COLUMNS:
        out[f"{variant}/{column}"] = table[column].to_numpy(dtype=np.float64)
    return out


def run_v1(case: Case) -> dict[str, np.ndarray]:
    """Evaluate, normalize and aggregate the predictions with the 1.6.1 code, via its CSV files.

    :param case: The shared case.
    :returns: Table arrays of every variant and the critical difference statistics of ``main``.
    """
    import pandas as pd
    from drevalpy.datasets.dataset import DrugResponseDataset
    from drevalpy.visualization.create_leaderboard import load_results
    from drevalpy.visualization.critical_difference_plot import CriticalDifferencePlot
    from drevalpy.visualization.utils import _normalize_metrics_by_mean_effects
    from scipy import stats

    from drevalpy.evaluation import AVAILABLE_METRICS, evaluate

    out: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        long = _tables(case, variant)
        long["rand_setting"] = "predictions"
        long["test_mode"] = "LCO"
        long["pubchem_id"] = long["drug_name"]
        rows = {}
        for (algorithm, split), part in long.groupby(["algorithm", "CV_split"], sort=False):
            dataset = DrugResponseDataset(
                response=part["y_true"].to_numpy(),
                cell_line_ids=part["cell_line_name"].to_numpy(),
                drug_ids=part["drug_name"].to_numpy(),
                predictions=part["y_pred"].to_numpy(),
            )
            rows[f"{algorithm}_predictions_LCO_split_{split}"] = {
                "algorithm": algorithm,
                "rand_setting": "predictions",
                "test_mode": "LCO",
                "CV_split": split,
                **evaluate(dataset, list(AVAILABLE_METRICS.keys())),
            }
        evaluation = pd.DataFrame.from_dict(rows, orient="index")
        evaluation = _normalize_metrics_by_mean_effects(evaluation_results=evaluation, true_vs_pred=long)
        with tempfile.TemporaryDirectory() as directory:
            evaluation.to_csv(Path(directory) / "evaluation_results.csv")
            long.to_csv(Path(directory) / "true_vs_pred.csv", index=False)
            table = load_results(
                str(Path(directory) / "evaluation_results.csv"),
                test_mode="LCO",
                true_vs_pred_path=str(Path(directory) / "true_vs_pred.csv"),
            )
        out |= _table_arrays(variant, table)
        if variant == "main":
            frame = evaluation[["algorithm", "CV_split", "Pearson: normalized"]]
            plot = CriticalDifferencePlot(eval_results_preds=frame, metric="Pearson: normalized")
            plot._draw()
            pivot = frame.pivot_table(index="CV_split", columns="algorithm", values="Pearson: normalized")
            out["main/cd/friedman_p"] = np.array([stats.friedmanchisquare(*[pivot[c] for c in pivot]).pvalue])
            out["main/cd/ranks"] = pivot.rank(ascending=False, axis=1).mean(axis=0).to_numpy(dtype=np.float64)
            out["main/cd/conover"] = plot.test_results.reindex(index=pivot.columns, columns=pivot.columns).to_numpy(
                dtype=np.float64
            )
    return out


def run_v2(case: Case) -> dict[str, np.ndarray]:
    """Build RunResults from the same predictions, normalize them and call the v2 leaderboard table.

    :param case: The shared case.
    :returns: Arrays keyed like :func:`run_v1`.
    """
    import scikit_posthocs as sp
    from scipy import stats

    from drevalpy.evaluation import score_predictions
    from drevalpy.types.results.experiment import ExperimentResult
    from drevalpy.types.results.run import RunResult
    from drevalpy.visualization.plots._leaderboard_table import build_leaderboard_df
    from drevalpy.visualization.plots.critical_difference import _build_cd_df

    out: dict[str, np.ndarray] = {}
    for variant, setting in VARIANTS.items():
        runs = []
        for index, algorithm in enumerate(setting["algorithms"]):
            for fold in range(N_SPLITS):
                mask = case.arrays[f"{variant}/fold"] == fold
                predictions = case.arrays[f"{variant}/y_pred"][index][mask]
                truth = case.arrays[f"{variant}/y_true"][mask]
                runs.append(
                    RunResult(
                        model_name=algorithm,
                        dataset_name="Leaderboard",
                        fold_index=fold,
                        predictions=predictions,
                        ground_truth=truth,
                        sample_ids=case.arrays[f"{variant}/sample"][mask].astype(object),
                        drug_ids=case.arrays[f"{variant}/drug"][mask].astype(object),
                        split_mode="LCO",
                        fold_id=f"split_{fold}",
                        metrics=score_predictions(predictions, truth, empty_as_dict=True),
                    )
                )
        normalized = ExperimentResult(runs).normalize(REFERENCE)
        out |= _table_arrays(variant, build_leaderboard_df(normalized))
        if variant == "main":
            frame = _build_cd_df(normalized, CD_METRIC)
            pivot = frame.pivot_table(index="CV_split", columns="algorithm", values=CD_METRIC)
            out["main/cd/friedman_p"] = np.array([stats.friedmanchisquare(*[pivot[c] for c in pivot]).pvalue])
            out["main/cd/ranks"] = pivot.rank(ascending=False, axis=1).mean(axis=0).to_numpy(dtype=np.float64)
            out["main/cd/conover"] = sp.posthoc_conover_friedman(pivot, p_adjust="fdr_bh").to_numpy(dtype=np.float64)
    return out

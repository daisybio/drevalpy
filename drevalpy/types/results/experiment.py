"""Experiment-level result: aggregates ModelResults across models."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from upath import UPath as Path

from drevalpy.evaluation import AVAILABLE_METRICS, _compute_metric_value
from drevalpy.log import get_logger
from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult

if TYPE_CHECKING:
    from drevalpy.visualization.requirements import PlotRequirement

logger = get_logger(__name__)


class ExperimentResult:
    """Groups all model results for a complete experiment."""

    def __init__(self, run_results: list[RunResult]) -> None:
        """Create an ExperimentResult from a flat list of RunResults.

        :param run_results: Non-empty list of RunResult objects sharing the same dataset.
        :raises ValueError: If the list is empty or contains inconsistent metadata.
        """
        if not run_results:
            raise ValueError("run_results must not be empty")

        dataset_names = {r.dataset_name for r in run_results}
        if len(dataset_names) > 1:
            raise ValueError(f"All RunResults must share the same dataset_name, got: {dataset_names}")

        split_modes = {r.split_mode for r in run_results if r.split_mode}
        if len(split_modes) > 1:
            raise ValueError(f"All RunResults must share the same split_mode, got: {split_modes}")

        self.dataset_name: str = dataset_names.pop()
        self.split_mode: str = split_modes.pop() if split_modes else ""
        self.normalized_by: str | None = None

        grouped: dict[str, list[RunResult]] = defaultdict(list)
        for r in run_results:
            grouped[r.model_name].append(r)

        self.models: list[ModelResult] = [
            ModelResult(model_name=name, dataset_name=self.dataset_name, runs=runs) for name, runs in grouped.items()
        ]

    @property
    def model_names(self) -> list[str]:
        """Names of all models in this experiment."""
        return [m.model_name for m in self.models]

    @property
    def has_randomization(self) -> bool:
        """Whether any run has randomization data."""
        return any(r.randomization is not None for m in self.models for r in m.runs)

    @property
    def has_robustness(self) -> bool:
        """Whether any run has robustness trial metadata."""
        return any("robustness_trial" in r.fold_metadata for m in self.models for r in m.runs)

    @property
    def n_models(self) -> int:
        """Number of distinct models."""
        return len(self.models)

    @property
    def max_folds(self) -> int:
        """Maximum number of folds across models."""
        return max((m.n_folds for m in self.models), default=0)

    def satisfies(self, requirements: frozenset[PlotRequirement]) -> bool:
        """Check if this experiment has data needed for a set of plot requirements.

        :param requirements: Set of requirements to check.
        :returns: True if all requirements are satisfied.
        """
        from drevalpy.visualization.requirements import PlotRequirement

        for req in requirements:
            if req == PlotRequirement.MULTIPLE_MODELS and self.n_models < 2:
                return False
            if req == PlotRequirement.MULTIPLE_FOLDS and self.max_folds < 2:
                return False
            if req == PlotRequirement.RANDOMIZATION and not self.has_randomization:
                return False
            if req == PlotRequirement.ROBUSTNESS and not self.has_robustness:
                return False
        return True

    @property
    def summary_table(self) -> dict[str, dict[str, float]]:
        """Mean metric per model: {model_name: {metric: mean}}.

        :returns: Nested dict suitable for DataFrame construction.
        """
        return {m.model_name: {k: v["mean"] for k, v in m.aggregate_metrics.items()} for m in self.models}

    def normalize(self, reference_model: str = "NaiveMeanEffectsPredictor") -> ExperimentResult:
        """Return a new ExperimentResult with metrics normalized against a reference model.

        :param reference_model: Name of the model to normalize against.
        :returns: A new ExperimentResult containing only non-reference runs with recomputed metrics.
        :raises ValueError: If already normalized or reference model not found.
        """
        if self.normalized_by is not None:
            raise ValueError(f"Already normalized by {self.normalized_by!r}")

        ref_runs_by_fold: dict[str, RunResult] = {}
        other_runs: list[RunResult] = []

        for model in self.models:
            if model.model_name == reference_model:
                for run in model.runs:
                    ref_runs_by_fold[run.fold_id] = run
            else:
                other_runs.extend(model.runs)

        if not ref_runs_by_fold:
            raise ValueError(f"Reference model {reference_model!r} not found. Available: {self.model_names}")

        normalized_runs: list[RunResult] = []
        for run in other_runs:
            if run.fold_id not in ref_runs_by_fold:
                raise ValueError(f"No reference run found for fold_id {run.fold_id!r}")
            ref_run = ref_runs_by_fold[run.fold_id]
            normalized_runs.append(_normalize_run(run, ref_run))

        result = ExperimentResult(normalized_runs)
        result.normalized_by = reference_model
        return result

    def save(self, directory: str | Path) -> None:
        """Save to a directory tree.

        :param directory: Root output directory.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "split_mode": self.split_mode,
            "normalized_by": self.normalized_by,
            "models": self.model_names,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

        for model_result in self.models:
            model_result.save(out / model_result.model_name)

    @classmethod
    def load(cls, directory: str | Path, *, with_trials: bool = True) -> ExperimentResult:
        """Load from a directory tree saved by ``save()``.

        :param directory: Root experiment directory.
        :param with_trials: Forwarded to :meth:`ModelResult.load`; pass ``False`` to skip
            reading the HPO trial predictions, which no visualization consumes.
        :returns: Reconstructed ExperimentResult.
        """
        path = Path(directory)
        meta = json.loads((path / "metadata.json").read_text())

        model_results = [ModelResult.load(path / name, with_trials=with_trials) for name in meta["models"]]

        all_runs: list[RunResult] = []
        for mr in model_results:
            for r in mr.runs:
                if not r.split_mode:
                    r.split_mode = meta.get("split_mode", "")
                all_runs.append(r)

        experiment = cls(all_runs)
        experiment.normalized_by = meta.get("normalized_by")
        _log_load_summary(experiment, all_runs, with_trials=with_trials)
        return experiment

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "ExperimentResult",
            f"    Dataset: {self.dataset_name}",
            f"    Split mode: {self.split_mode}",
            f"    Normalized by: {self.normalized_by}",
            f"    Models: {len(self.models)}",
        ]
        for m in self.models:
            agg = m.aggregate_metrics
            metric_str = ", ".join(f"{k}={v['mean']:.4f}" for k, v in agg.items()) if agg else "no metrics"
            lines.append(f"        {m.model_name} ({m.n_folds} folds): {metric_str}")
        return "\n".join(lines)


def _run_array_bytes(run: RunResult) -> int:
    """Approximate the heap cost of one run's arrays.

    Object-dtype id arrays only report their pointer table via ``nbytes``, so the shared
    strings behind them are added once each - which is the whole point of interning them.

    :param run: Run to measure.
    :returns: Approximate number of bytes retained by the run's arrays.
    """
    total = run.predictions.nbytes + run.ground_truth.nbytes
    for ids in (run.cell_line_ids, run.drug_ids):
        total += ids.nbytes
        if ids.dtype == object:
            total += sum(sys.getsizeof(value) for value in set(ids.tolist()))
    if run.trials:
        total += sum(trial.predictions.nbytes for trial in run.trials)
    return total


def _log_load_summary(experiment: ExperimentResult, runs: list[RunResult], *, with_trials: bool) -> None:
    """Emit the single line that establishes the scale of a loaded experiment."""
    rows = sum(len(r.predictions) for r in runs)
    total_bytes = sum(_run_array_bytes(r) for r in runs)
    logger.info(
        "Loaded ExperimentResult %r: %d models, %d runs, %d prediction rows, %.2f GB of arrays (trials %s)",
        experiment.dataset_name,
        experiment.n_models,
        len(runs),
        rows,
        total_bytes / 1024**3,
        "loaded" if with_trials else "skipped",
    )


def _normalize_run(run: RunResult, ref_run: RunResult) -> RunResult:
    """Normalize a single RunResult against a reference RunResult."""
    ref_index = pd.MultiIndex.from_arrays(
        [np.asarray(ref_run.cell_line_ids, dtype=object), np.asarray(ref_run.drug_ids, dtype=object)]
    )
    ref_predictions = np.asarray(ref_run.predictions)
    if ref_index.has_duplicates:
        # The dict lookup this replaced let the last occurrence of a pair win.
        keep = ~ref_index.duplicated(keep="last")
        ref_index = ref_index[keep]
        ref_predictions = ref_predictions[keep]
    positions = ref_index.get_indexer(
        pd.MultiIndex.from_arrays([np.asarray(run.cell_line_ids, dtype=object), np.asarray(run.drug_ids, dtype=object)])
    )
    # get_indexer yields -1 for pairs the reference never predicted; those normalize
    # against 0.0, matching the dict-lookup default this replaced.
    ref_preds = np.where(positions >= 0, ref_predictions[positions.clip(min=0)], 0.0)

    norm_gt = run.ground_truth - ref_preds
    norm_pred = run.predictions - ref_preds

    valid = ~np.isnan(norm_pred) & ~np.isnan(norm_gt)
    metrics: dict[str, float] = {}
    if valid.any():
        for metric_name in AVAILABLE_METRICS:
            metrics[metric_name] = _compute_metric_value(metric_name, norm_pred[valid], norm_gt[valid])

    normalized_trials = None
    if run.trials:
        normalized_trials = [
            TrialResult(
                hyperparameters=trial.hyperparameters,
                metrics=trial.metrics,
                optimization_metric=trial.optimization_metric,
                predictions=trial.predictions,
            )
            for trial in run.trials
        ]

    return RunResult(
        model_name=run.model_name,
        dataset_name=run.dataset_name,
        split_mode=run.split_mode,
        fold_index=run.fold_index,
        fold_id=run.fold_id,
        predictions=norm_pred,
        ground_truth=norm_gt,
        cell_line_ids=run.cell_line_ids,
        drug_ids=run.drug_ids,
        best_hyperparameters=run.best_hyperparameters,
        metrics=metrics,
        fold_metadata=run.fold_metadata,
        trials=normalized_trials,
        randomization=run.randomization,
    )

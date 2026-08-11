"""Experiment-level result: aggregates ModelResults across models."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from upath import UPath as Path

from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.visualization.plot_requirements import PlotRequirement


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
        from drevalpy.visualization.plot_requirements import PlotRequirement

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

    def save(self, directory: str | Path) -> None:
        """Save to a directory tree.

        :param directory: Root output directory.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "split_mode": self.split_mode,
            "models": self.model_names,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

        for model_result in self.models:
            model_result.save(out / model_result.model_name)

    @classmethod
    def load(cls, directory: str | Path) -> ExperimentResult:
        """Load from a directory tree saved by ``save()``.

        :param directory: Root experiment directory.
        :returns: Reconstructed ExperimentResult.
        """
        path = Path(directory)
        meta = json.loads((path / "metadata.json").read_text())

        model_results = [ModelResult.load(path / name) for name in meta["models"]]

        all_runs: list[RunResult] = []
        for mr in model_results:
            for r in mr.runs:
                if not r.split_mode:
                    r.split_mode = meta.get("split_mode", "")
                all_runs.append(r)

        return cls(all_runs)

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "ExperimentResult",
            f"    Dataset: {self.dataset_name}",
            f"    Split mode: {self.split_mode}",
            f"    Models: {len(self.models)}",
        ]
        for m in self.models:
            agg = m.aggregate_metrics
            metric_str = ", ".join(f"{k}={v['mean']:.4f}" for k, v in agg.items()) if agg else "no metrics"
            lines.append(f"        {m.model_name} ({m.n_folds} folds): {metric_str}")
        return "\n".join(lines)

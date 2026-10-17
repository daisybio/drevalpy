"""Model-level result: aggregates RunResults across folds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from upath import UPath as Path

from drevalpy.types.results.run import RunResult


@dataclass
class ModelResult:
    """Groups all fold results for a single model on a single dataset."""

    model_name: str
    dataset_name: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        """Number of distinct folds, independent of diagnostic variants."""
        return len({run.fold_key for run in self.runs})

    @property
    def n_runs(self) -> int:
        """Number of executions including diagnostic variants."""
        return len(self.runs)

    def select_runs(
        self,
        *,
        randomization: tuple[str, str] | Literal["all"] | None = None,
        robustness: int | Literal["all"] | None = None,
    ) -> list[RunResult]:
        """Select executions, defaulting to ordinary runs only.

        :param randomization: Exact setting, ``"all"``, or ``None`` for unrandomized data.
        :param robustness: Exact trial, ``"all"``, or ``None`` for ordinary pair ordering.
        :returns: Matching runs in their original order.
        """
        return [
            run
            for run in self.runs
            if (randomization == "all" or run.randomization == randomization)
            and (robustness == "all" or run.robustness_trial == robustness)
        ]

    @property
    def aggregate_metrics(self) -> dict[str, dict[str, float]]:
        """Mean and std of each metric across folds.

        :returns: Mapping of metric_name -> {"mean": ..., "std": ...}.
        """
        if not self.runs:
            return {}
        all_metrics: dict[str, list[float]] = {}
        for run in self.select_runs():
            for key, value in run.metrics.items():
                all_metrics.setdefault(key, []).append(value)
        return {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values))} for key, values in all_metrics.items()
        }

    def save(self, directory: str | Path) -> None:
        """Save to a directory with metadata.json and one .npz per fold.

        :param directory: Output directory path.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "n_folds": self.n_folds,
            "n_runs": self.n_runs,
            "aggregate_metrics": self.aggregate_metrics,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))

        for i, run in enumerate(self.runs):
            run.save(str(out / f"fold_{i}.npz"))

    @classmethod
    def load(cls, directory: str | Path, *, with_trials: bool = True) -> ModelResult:
        """Load from a directory saved by ``save()``.

        :param directory: Path to the model result directory.
        :param with_trials: Forwarded to :meth:`RunResult.load`; pass ``False`` to skip
            reading the HPO trial predictions.
        :returns: Reconstructed ModelResult.
        """
        path = Path(directory)
        meta = json.loads((path / "metadata.json").read_text())

        fold_files = (
            [path / f"fold_{i}.npz" for i in range(meta["n_runs"])]
            if "n_runs" in meta
            else sorted(path.glob("fold_*.npz"))
        )
        runs = [RunResult.load(str(f), with_trials=with_trials) for f in fold_files]

        return cls(
            model_name=meta["model_name"],
            dataset_name=meta["dataset_name"],
            runs=runs,
        )

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "ModelResult",
            f"    Model: {self.model_name}",
            f"    Dataset: {self.dataset_name}",
            f"    Folds: {self.n_folds}",
        ]
        agg = self.aggregate_metrics
        if agg:
            lines.append("    Metrics (mean +/- std):")
            for key, stats in agg.items():
                lines.append(f"        {key}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
        return "\n".join(lines)

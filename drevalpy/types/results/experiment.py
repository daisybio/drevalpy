"""Experiment-level result: aggregates ModelResults across models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from upath import UPath as Path

from drevalpy.types.results.model import ModelResult


@dataclass
class ExperimentResult:
    """Groups all model results for a complete experiment."""

    dataset_name: str
    split_mode: str
    models: list[ModelResult] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        """Names of all models in this experiment."""
        return [m.model_name for m in self.models]

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

        models = [ModelResult.load(path / name) for name in meta["models"]]

        return cls(
            dataset_name=meta["dataset_name"],
            split_mode=meta["split_mode"],
            models=models,
        )

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

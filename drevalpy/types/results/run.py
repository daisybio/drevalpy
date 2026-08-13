"""Run result dataclass."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from upath import UPath as Path

from drevalpy.log import get_logger

from .trial import TrialResult

logger = get_logger(__name__)


@dataclass
class RunResult:
    """Output of a single Run."""

    model_name: str
    dataset_name: str
    fold_index: int
    predictions: np.ndarray
    ground_truth: np.ndarray
    cell_line_ids: np.ndarray
    drug_ids: np.ndarray
    split_mode: str = ""
    fold_id: str = ""
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metadata: dict[str, Any] = field(default_factory=dict)
    trials: list[TrialResult] | None = None
    randomization: tuple[str, str] | None = None

    def __repr__(self) -> str:
        """Formatted summary."""
        lines = [
            "RunResult",
            f"    Model: {self.model_name}",
            f"    Dataset: {self.dataset_name}",
        ]

        if self.randomization:
            lines.append(f"        Randomization: {self.randomization[0]} ({self.randomization[1]})")
        else:
            lines.append("        Randomization: None")

        lines.append(f"    Fold: {self.fold_index}")

        for k, v in self.fold_metadata.items():
            if k != "fold_index":
                lines.append(f"        {k}: {v}")

        lines.append(f"    Predictions: {len(self.predictions)} pairs")
        lines.append(f"    Ground truth: {int(np.sum(~np.isnan(self.ground_truth)))} non-NaN values")

        if self.best_hyperparameters:
            lines.append("    Hyperparameters:")
            for k, v in self.best_hyperparameters.items():
                lines.append(f"        {k}: {v}")

        if self.metrics:
            lines.append("    Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"        {k}: {v:.4f}")

        if self.trials:
            lines.append(f"    HPO Trials: {len(self.trials)}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz file.

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "predictions": self.predictions,
            "ground_truth": self.ground_truth,
            "cell_line_ids": np.asarray(self.cell_line_ids, dtype=str),
            "drug_ids": np.asarray(self.drug_ids, dtype=str),
        }
        trials_meta = None
        if self.trials:
            trials_meta = []
            for i, t in enumerate(self.trials):
                arrays[f"trial_{i}_predictions"] = t.predictions
                trials_meta.append(
                    {
                        "hyperparameters": t.hyperparameters,
                        "metrics": t.metrics,
                        "optimization_metric": t.optimization_metric,
                    }
                )
        meta = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "split_mode": self.split_mode,
            "fold_index": self.fold_index,
            "fold_id": self.fold_id,
            "best_hyperparameters": self.best_hyperparameters,
            "metrics": self.metrics,
            "fold_metadata": self.fold_metadata,
            "randomization": list(self.randomization) if self.randomization else None,
            "trials": trials_meta,
        }
        arrays["_metadata"] = np.array(json.dumps(meta))
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path, *, with_trials: bool = True) -> RunResult:
        """Load from a .npz file saved by ``save()``.

        :param path: Path to the .npz file.
        :param with_trials: Whether to read the ``trial_*_predictions`` arrays. These are
            typically an order of magnitude larger than the fold's own predictions and no
            visualization reads them, so the report path opts out. ``np.load`` is lazy, so
            skipping them means they are never read off disk.
        :returns: Reconstructed RunResult.
        """
        logger.debug("Loading run %s (with_trials=%s)", path, with_trials)
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["_metadata"]))
        trials = None
        if with_trials and meta.get("trials"):
            trials = [
                TrialResult(
                    hyperparameters=t["hyperparameters"],
                    metrics=t["metrics"],
                    optimization_metric=t["optimization_metric"],
                    predictions=np.asarray(data[f"trial_{i}_predictions"]),
                )
                for i, t in enumerate(meta["trials"])
            ]
        return cls(
            model_name=meta["model_name"],
            dataset_name=meta["dataset_name"],
            split_mode=meta.get("split_mode", ""),
            fold_index=meta["fold_index"],
            fold_id=meta.get("fold_id", ""),
            predictions=np.asarray(data["predictions"]),
            ground_truth=np.asarray(data["ground_truth"]),
            cell_line_ids=intern_ids(data["cell_line_ids"]),
            drug_ids=intern_ids(data["drug_ids"]),
            best_hyperparameters=meta.get("best_hyperparameters", {}),
            metrics=meta.get("metrics", {}),
            fold_metadata=meta.get("fold_metadata", {}),
            trials=trials,
            randomization=tuple(meta["randomization"]) if meta.get("randomization") else None,
        )


def intern_ids(ids: np.ndarray) -> np.ndarray:
    """Re-express a fixed-width unicode id array as an object array of shared strings.

    NumPy stores ``<U40`` at 160 bytes per element regardless of the actual id length, and
    since the ids repeat heavily across a fold this dominates the on-heap size of a loaded
    experiment. An object array of deduplicated Python ``str`` costs one pointer per element
    plus one string per distinct id. Indexing still yields a ``str``, and ``save()``
    normalises with ``dtype=str``, so the round-trip is unchanged.

    :param ids: Array of entity ids, typically ``<U*`` as read from an npz.
    :returns: Object-dtype array of the same shape holding shared ``str`` objects.
    """
    unique, inverse = np.unique(ids, return_inverse=True)
    shared = np.empty(len(unique), dtype=object)
    shared[:] = [str(value) for value in unique]
    return shared[inverse.reshape(ids.shape)]

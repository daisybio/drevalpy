"""Model-level result: aggregates RunResults across folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from upath import UPath as Path

from drevalpy.errors import InvalidInputError
from drevalpy.evaluation import Metric, evaluate_per_group
from drevalpy.evaluation._grouping import group_labeller
from drevalpy.evaluation._scoring import mean_of_defined
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.evaluation._grouping import GroupBy
    from drevalpy.types.data.dataset import Dataset


@dataclass
class ModelResult:
    """Groups all fold results for a single model on a single dataset."""

    model_name: str
    dataset_name: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        """Number of distinct folds, independent of stress-test variants."""
        return len({run.fold_key for run in self.runs})

    @property
    def n_runs(self) -> int:
        """Number of runs including stress-test variants."""
        return len(self.runs)

    def select_runs(
        self,
        *,
        randomization: tuple[str, str] | Literal["all"] | None = None,
        robustness: int | Literal["all"] | None = None,
    ) -> list[RunResult]:
        """Select runs, defaulting to ordinary runs only.

        :param randomization: Exact setting, ``"all"``, or ``None`` for unrandomized data.
        :param robustness: Exact repeat, ``"all"``, or ``None`` for ordinary pair ordering.
        :returns: Matching runs in their original order.
        """
        return [
            run
            for run in self.runs
            if (randomization == "all" or run.randomization == randomization)
            and (robustness == "all" or run.robustness_repeat == robustness)
        ]

    @property
    def aggregate_metrics(self) -> dict[str, dict[str, float]]:
        """Mean and sample standard deviation (``ddof=1``, ``0`` for a single fold) of each metric across folds.

        :returns: Mapping of metric_name -> {"mean": ..., "std": ...}.
        """
        if not self.runs:
            return {}
        all_metrics: dict[str, list[float]] = {}
        for run in self.select_runs():
            for key, value in run.metrics.items():
                all_metrics.setdefault(key, []).append(value)
        return {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            for key, values in all_metrics.items()
        }

    def aggregate_group_metrics(self, group_by: GroupBy, dataset: Dataset | None = None) -> dict[str, dict[str, float]]:
        """Mean per-group score of each metric within every group of pairs, averaged across folds.

        Each metric of :attr:`aggregate_metrics` that is a registered evaluation metric is scored per run within
        each group of pairs, averaged over the groups where it is defined, and then averaged over the runs where any
        group was defined. :doc:`/reference/metrics` defines the groups and when a score is undefined (``NaN``).

        Raises ``ValueError`` if ``group_by`` is neither built in nor an unambiguous annotation column of
        ``dataset``, or needs a ``dataset`` that was not given.

        :param group_by: ``"drug"``, ``"sample"`` or ``"study"``, or a sample or drug annotation column of
            ``dataset`` such as ``"tissue"`` or ``"ncit_class"`` (``"sample:<column>"`` / ``"drug:<column>"``
            pick the axis). A pair whose annotation is missing, empty or ``"unknown"`` belongs to no group.
        :param dataset: Dataset whose annotations label the pairs; needed only for an annotation grouping.
        :returns: Mapping of metric_name -> {"mean": ..., "n_groups": ...}, where ``n_groups``
            is the mean number of groups with a defined score per run.
        """
        labels_of = group_labeller(group_by, dataset)
        known = {metric.value for metric in Metric}
        per_run: dict[str, list[tuple[float, int]]] = {}
        for run in self.select_runs():
            names = [name for name in run.metrics if name in known]
            if not names:
                continue
            scores = evaluate_per_group(run.predictions, run.ground_truth, names, labels=labels_of(run))
            n_defined = scores.n_defined()
            for name, mean in scores.mean().items():
                per_run.setdefault(name, []).append((mean, n_defined[name]))
        return {
            name: {
                "mean": mean_of_defined([mean for mean, _ in results]),
                "n_groups": float(np.mean([count for _, count in results])),
            }
            for name, results in per_run.items()
        }

    def save(self, directory: str | Path) -> None:
        """Save to a directory with one .npz per run, named ``fold_<i>.npz`` in run order.

        The ``fold_<i>.npz`` files of an earlier save to the same directory are removed first, since
        :meth:`load` reads every one it finds.

        :param directory: Output directory path.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("fold_*.npz"):
            stale.unlink()
        for i, run in enumerate(self.runs):
            run.save(out / f"fold_{i}.npz")

    @classmethod
    def load(cls, directory: str | Path, *, with_trials: bool = True) -> ModelResult:
        """Load from a directory saved by ``save()``; the model and dataset names are those of its runs.

        :param directory: Path to the model result directory.
        :param with_trials: Forwarded to :meth:`RunResult.load`; pass ``False`` to skip
            reading the HPO trial predictions.
        :returns: Reconstructed ModelResult.
        :raises InvalidInputError: If the directory holds no ``fold_<i>.npz`` file.
        """
        path = Path(directory)
        fold_files = sorted(path.glob("fold_*.npz"), key=lambda file: int(file.stem.removeprefix("fold_")))
        if not fold_files:
            raise InvalidInputError(f"{path} holds no fold_<i>.npz file")
        runs = [RunResult.load(file, with_trials=with_trials) for file in fold_files]
        return cls(model_name=runs[0].model_name, dataset_name=runs[0].dataset_name, runs=runs)

    def __repr__(self) -> str:
        """Return a formatted summary."""
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

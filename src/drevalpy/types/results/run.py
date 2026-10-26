"""Run result dataclass."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from upath import UPath as Path

from drevalpy.log import get_logger
from drevalpy.types.enums.randomization import randomization_label

from ._identity import RunIdentity
from .trial import TrialResult

logger = get_logger(__name__)


def _indented_items(
    items: Mapping[str, Any],
    format_value: Callable[[Any], str] = str,
) -> list[str]:
    """Render *items* as the ``__repr__``'s inner-level ``key: value`` lines.

    :param items: Mapping to render, in iteration order.
    :param format_value: Applied to each value; defaults to ``str``.
    :returns: One line per entry, indented to the nested level.
    """
    return [f"        {key}: {format_value(value)}" for key, value in items.items()]


def _section(
    heading: str,
    items: Mapping[str, Any],
    format_value: Callable[[Any], str] = str,
) -> list[str]:
    """Render a headed block of ``key: value`` lines, or nothing when *items* is empty.

    :param heading: Section heading, already indented.
    :param items: Mapping to render under the heading.
    :param format_value: Applied to each value; defaults to ``str``.
    :returns: The heading followed by its entries, or an empty list.
    """
    if not items:
        return []
    return [heading, *_indented_items(items, format_value)]


@dataclass
class RunResult:
    """Output of a single Run.

    ``raw_metrics`` and ``raw_per_drug_pearson`` are empty and ``None`` until
    :meth:`~drevalpy.types.results.ExperimentResult.normalize` replaces ``predictions``,
    ``ground_truth`` and ``metrics`` by their residuals against the reference model. They keep
    what the leaderboard still needs from before that: the pooled ``metrics`` and the mean over
    drugs of the per-drug Pearson, both of the unnormalized predictions.
    """

    model_name: str
    dataset_name: str
    fold_index: int
    predictions: np.ndarray
    ground_truth: np.ndarray
    sample_ids: np.ndarray
    drug_ids: np.ndarray
    split_mode: str = ""
    fold_id: str = ""
    best_hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metadata: dict[str, Any] = field(default_factory=dict)
    trials: list[TrialResult] | None = None
    randomization: tuple[str, str] | None = None
    raw_metrics: dict[str, float] = field(default_factory=dict)
    raw_per_drug_pearson: float | None = None

    @property
    def robustness_repeat(self) -> int | None:
        """Robustness repeat, or ``None`` for ordinary pair ordering."""
        return self.fold_metadata.get("robustness_repeat")

    @property
    def identity(self) -> RunIdentity:
        """Immutable identity shared with experiment execution."""
        return RunIdentity.from_metadata(
            self.model_name,
            self.dataset_name,
            self.split_mode,
            self.fold_index,
            self.fold_id,
            self.randomization,
            self.fold_metadata,
        )

    @property
    def fold_key(self) -> tuple[str, str, str, str | int]:
        """Dataset/split/fold identity shared by ordinary and stress-test runs."""
        return self.identity.fold_key

    @property
    def run_key(self) -> tuple:
        """Identity including stress-test settings and recorded provenance."""
        return self.identity.run_key

    @property
    def run_id(self) -> str:
        """Unambiguous serialized run identity for report data keys."""
        return json.dumps(self.run_key, separators=(",", ":"))

    def __repr__(self) -> str:
        """Return a formatted summary."""
        lines = [
            "RunResult",
            f"    Model: {self.model_name}",
            f"    Dataset: {self.dataset_name}",
            f"        Randomization: {randomization_label(self.randomization) if self.randomization else None}",
            f"    Fold: {self.fold_index}",
            *_indented_items(self._unique_fold_metadata()),
            f"    Predictions: {len(self.predictions)} pairs",
            f"    Ground truth: {int(np.sum(~np.isnan(self.ground_truth)))} non-NaN values",
            *_section("    Hyperparameters:", self.best_hyperparameters),
            *_section("    Metrics:", self.metrics, format_value="{:.4f}".format),
        ]
        if self.trials:
            lines.append(f"    HPO Trials: {len(self.trials)}")
        return "\n".join(lines)

    def _unique_fold_metadata(self) -> dict[str, Any]:
        """Return the fold metadata that no field of this run already holds.

        :returns: ``fold_metadata`` without the fold fields (``fold_index``, ``fold_id``, ``split_mode``) and the
            dataset name, which the fields store.
        """
        return {k: v for k, v in self.fold_metadata.items() if k not in _FOLD_METADATA_DUPLICATES}

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz file.

        :param path: Output file path (should end in .npz).
        """
        arrays: dict[str, np.ndarray] = {
            "predictions": self.predictions,
            "ground_truth": self.ground_truth,
            "sample_ids": np.asarray(self.sample_ids, dtype=str),
            "drug_ids": np.asarray(self.drug_ids, dtype=str),
        }
        trials_meta = None
        if self.trials:
            trials_meta = []
            for i, trial in enumerate(self.trials):
                arrays[f"trial_{i}_predictions"] = trial.predictions
                trials_meta.append({name: getattr(trial, name) for name in _TRIAL_META_FIELDS})
        meta = {name: getattr(self, name) for name in _META_FIELDS}
        meta["fold_metadata"] = self._unique_fold_metadata()
        meta["trials"] = trials_meta
        arrays["_metadata"] = np.array(json.dumps(meta))
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)

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
        with Path(path).open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            meta = json.loads(str(data["_metadata"]))
            trials_meta = meta.pop("trials")
            randomization = meta.pop("randomization")
            trials = None
            if with_trials and trials_meta:
                trials = [
                    TrialResult(**trial_meta, predictions=np.asarray(data[f"trial_{i}_predictions"]))
                    for i, trial_meta in enumerate(trials_meta)
                ]
            return cls(
                predictions=np.asarray(data["predictions"]),
                ground_truth=np.asarray(data["ground_truth"]),
                sample_ids=_intern_ids(data["sample_ids"]),
                drug_ids=_intern_ids(data["drug_ids"]),
                randomization=tuple(randomization) if randomization else None,
                trials=trials,
                **meta,
            )


_ARRAY_FIELDS = ("predictions", "ground_truth", "sample_ids", "drug_ids")
_FOLD_METADATA_DUPLICATES = frozenset(f.name for f in fields(RunResult)) | {"dataset"}
_META_FIELDS = tuple(f.name for f in fields(RunResult) if f.name not in (*_ARRAY_FIELDS, "fold_metadata", "trials"))
_TRIAL_META_FIELDS = tuple(f.name for f in fields(TrialResult) if f.name != "predictions")


def _intern_ids(ids: np.ndarray) -> np.ndarray:
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

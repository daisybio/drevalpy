"""Immutable fit decisions made once, before HPO or model training begins."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from drevalpy.evaluation import Metric
from drevalpy.models.config.immutable import freeze_value, thaw_value
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import Validation, canonical, splitter_registry
from drevalpy.registry.splitter._validation import validate_folds
from drevalpy.types import Dataset, Fold, PairMask
from drevalpy.types.results._identity import RunIdentity


def _scope_provenance(scope: PairMask | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    return {
        "shape": list(scope.shape),
        "count": len(scope),
        "pair_seed": scope.pair_seed,
        "sha256": hashlib.sha256(scope.mask.tobytes()).hexdigest(),
    }


@dataclass(frozen=True)
class FitScopes:
    """Selections for tuning and final fitting, in their original pair order."""

    training: PairMask
    stopping: PairMask | None
    validation: PairMask
    final_training: PairMask
    test: PairMask

    def provenance(self) -> dict[str, Any]:
        """Return serializable scope checksums, counts, and pair-order seeds."""
        return {
            name: _scope_provenance(getattr(self, name))
            for name in (
                "training",
                "stopping",
                "validation",
                "final_training",
                "test",
            )
        }


@dataclass(frozen=True)
class ExecutionSpec:
    """All fold decisions consumed by one executor; no fitted state or dataset."""

    model: ModelSpec
    scopes: FitScopes
    identity: RunIdentity
    hyperparameter_tuning: bool
    hpo_metric: str
    hpo_num_trials: int
    hpo_random_state: int
    seed: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Capture independent, recursively frozen provenance."""
        object.__setattr__(self, "metadata", freeze_value(copy.deepcopy(dict(self.metadata))))

    def result_metadata(self) -> dict[str, Any]:
        """Return an independent JSON-compatible snapshot for results."""
        return thaw_value(self.metadata)


def build_execution(
    model: ModelSpec,
    dataset: Dataset,
    fold: Fold,
    *,
    hyperparameter_tuning: bool,
    hpo_metric: str,
    hpo_num_trials: int,
    hpo_random_state: int,
    seed: int,
) -> ExecutionSpec:
    """Validate the original fold and reserve stopping data exactly once.

    An unknown *hpo_metric* raises ``ValueError`` from ``Metric``.

    :param model: Named model recipe.
    :param dataset: Source data (not retained).
    :param fold: Original training/validation/test partition.
    :param hyperparameter_tuning: Whether to search parameters.
    :param hpo_metric: Selection metric.
    :param hpo_num_trials: Number of trials.
    :param hpo_random_state: Sampler seed.
    :param seed: Root seed from which every fit's RNG seed is derived.
    :returns: Immutable execution decisions.
    :raises TypeError: If *model* is not a ``ModelSpec``.
    :raises ValueError: If *hpo_num_trials* is negative.
    """
    if not isinstance(model, ModelSpec):
        raise TypeError("Experiment models must be ModelSpec values")
    Metric(hpo_metric)
    if hpo_num_trials < 0:
        raise ValueError(f"n_trials must be >= 0 (got {hpo_num_trials}); use 0 for default-only tuning")
    mode = canonical(fold.metadata.get("split_mode", ""))
    validation = splitter_registry.metadata(mode)["validation"] if mode in splitter_registry else mode
    validate_folds(
        [fold], cast(Validation, validation if validation in ("LSO", "LDO", "LTO", "LStO") else "LPO"), dataset
    )
    stopping, validation_scope = None, fold.val
    if model.supports_early_stopping and len(fold.val) > 1:
        stopping, validation_scope = fold.early_stopping_mask()
    scopes = FitScopes(fold.train, stopping, validation_scope, fold.train | validation_scope, fold.test)
    metadata = copy.deepcopy(fold.metadata)
    if "split_mode" in metadata:
        metadata["split_mode"] = mode
    if dataset.randomization_provenance is not None:
        metadata["randomization_provenance"] = copy.deepcopy(dataset.randomization_provenance)
    metadata["execution"] = {
        "config": model.config.model_dump(mode="json"),
        "scopes": scopes.provenance(),
        "hpo": {
            "enabled": hyperparameter_tuning,
            "metric": hpo_metric,
            "num_trials": hpo_num_trials,
            "random_state": hpo_random_state,
        },
        "seed": seed,
    }
    identity = RunIdentity.from_metadata(
        model.name,
        dataset.name,
        mode,
        metadata.get("fold_index", 0),
        metadata.get("fold_id", ""),
        dataset.randomization,
        metadata,
    )
    return ExecutionSpec(
        model,
        scopes,
        identity,
        hyperparameter_tuning,
        hpo_metric,
        hpo_num_trials,
        hpo_random_state,
        seed,
        metadata,
    )

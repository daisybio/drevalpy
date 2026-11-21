"""Immutable fit decisions made once, before HPO or model training begins."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from drevalpy.data.splitters._early_stopping import early_stopping_split
from drevalpy.errors import InvalidInputError
from drevalpy.evaluation import Metric
from drevalpy.models.config.immutable import freeze_value, thaw_value
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import Validation, canonical, splitter_registry
from drevalpy.registry.splitter._groups import MODES
from drevalpy.registry.splitter._validation import validate_folds
from drevalpy.types import Dataset, Fold, SelectionMask
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.results._identity import RunIdentity


def _scope_provenance(scope: SelectionMask | None) -> dict[str, Any] | None:
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

    training: SelectionMask
    stopping: SelectionMask | None
    validation: SelectionMask
    final_training: SelectionMask
    test: SelectionMask

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


def _validation_of(mode: str) -> Validation:
    """Return the leakage constraint of split *mode*; an unregistered mode is checked as its own name or LPO."""
    mode = canonical(mode)
    validation = splitter_registry.metadata(mode)["validation"] if mode in splitter_registry else mode
    return cast(Validation, validation if validation in MODES else "LPO")


def check_mode_supported(model: ModelSpec, mode: str) -> None:
    """Reject a model that cannot be scored under split *mode* before any work is done.

    A single-drug model fits one estimator per drug, so it has no prediction for the unseen
    drugs of a mode that holds drugs out (LDO).

    :param model: Named model recipe.
    :param mode: Split mode the model would be evaluated under, or an alias of one.
    :raises InvalidInputError: If *model* is a single-drug model and *mode* holds out drugs.
    """
    from drevalpy.registry.predictor import get as get_predictor

    if _validation_of(mode) == "LDO" and get_predictor(model.config.predictor.name).scope is ModelScope.SINGLE_DRUG:
        raise InvalidInputError(
            f"{model.name} is a single-drug model: it fits one model per drug and cannot predict the unseen "
            f"drugs of {canonical(mode)}; evaluate it under another split mode."
        )


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

    An unknown *hpo_metric* raises ``ValueError`` from ``Metric``. A single-drug model under LDO
    raises ``InvalidInputError`` from :func:`check_mode_supported`. A model that stops early gets
    whole groups of the split mode cut out of the validation set as its stopping set by
    :func:`~drevalpy.data.splitters._early_stopping.early_stopping_split`, which raises
    ``InvalidInputError`` when the validation set has fewer than two groups.

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
    :raises InvalidInputError: If the fold was split on another response measure than the dataset's.
    """
    if not isinstance(model, ModelSpec):
        raise TypeError("Experiment models must be ModelSpec values")
    Metric(hpo_metric)
    if hpo_num_trials < 0:
        raise ValueError(f"n_trials must be >= 0 (got {hpo_num_trials}); use 0 for default-only tuning")
    mode = canonical(fold.metadata.get("split_mode", ""))
    validation = _validation_of(mode)
    check_mode_supported(model, mode)
    validate_folds([fold], validation, dataset)
    stopping, validation_scope = None, fold.val
    if model.supports_early_stopping:
        stopping, validation_scope = early_stopping_split(dataset, fold, validation)
    scopes = FitScopes(fold.train, stopping, validation_scope, fold.train | validation_scope, fold.test)
    fold_measure = fold.metadata.get("measure", dataset.measure)
    if fold_measure != dataset.measure:
        msg = (
            f"The fold was split on the response measure {fold_measure!r}, but the dataset trains on "
            f"{dataset.measure!r}; split again, or use dataset.with_measure({fold_measure!r})"
        )
        raise InvalidInputError(msg)
    metadata = copy.deepcopy(fold.metadata)
    metadata["measure"] = dataset.measure
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

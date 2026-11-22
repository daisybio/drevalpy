"""Validate folds and capture run provenance before tuning or training."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, cast

from drevalpy.data.splitters._folds import early_stopping_split
from drevalpy.errors import InvalidInputError
from drevalpy.models.spec import ModelSpec
from drevalpy.registry.splitter import Validation, canonical, splitter_registry
from drevalpy.registry.splitter._groups import MODES
from drevalpy.registry.splitter._validation import validate_folds
from drevalpy.types import Dataset, Fold, SelectionMask
from drevalpy.types.enums import ModelScope


def _scope_provenance(scope: SelectionMask | None) -> dict[str, Any] | None:
    if scope is None:
        return None
    return {
        "shape": list(scope.shape),
        "count": len(scope),
        "pair_seed": scope.pair_seed,
        "sha256": hashlib.sha256(scope.mask.tobytes()).hexdigest(),
    }


def _fold_provenance(fold: Fold) -> dict[str, Any]:
    """Return serializable scope checksums, counts, and pair-order seeds for *fold*."""
    return {
        "training": _scope_provenance(fold.train),
        "stopping": _scope_provenance(fold.stopping),
        "validation": _scope_provenance(fold.val),
        "final_training": _scope_provenance(fold.train_val),
        "test": _scope_provenance(fold.test),
    }


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


def prepare_fold(model: ModelSpec, dataset: Dataset, fold: Fold, mode: str) -> Fold:
    """Validate the original fold and reserve stopping data for models that use it.

    :param model: Named model recipe.
    :param dataset: Dataset the fold indexes.
    :param fold: Original training, validation, and test partition.
    :param mode: Canonical split mode.
    :returns: Fold with the validation and stopping masks used during fitting.
    :raises InvalidInputError: If the fold was split on another response measure than the dataset's.
    """
    validation = _validation_of(mode)
    check_mode_supported(model, mode)
    validate_folds([fold], validation, dataset)
    stopping, validation_scope = None, fold.val
    if model.supports_early_stopping:
        stopping, validation_scope = early_stopping_split(dataset, fold, validation)

    fold_measure = fold.metadata.get("measure", dataset.measure)
    if fold_measure != dataset.measure:
        msg = (
            f"The fold was split on the response measure {fold_measure!r}, but the dataset trains on "
            f"{dataset.measure!r}; split again, or use dataset.with_measure({fold_measure!r})"
        )
        raise InvalidInputError(msg)
    return Fold(
        train=fold.train,
        test=fold.test,
        val=validation_scope,
        metadata=fold.metadata,
        stopping=stopping,
    )


def result_metadata(
    model: ModelSpec, dataset: Dataset, fold: Fold, mode: str, *, hpo: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Capture independent result metadata before training.

    :param model: Named model recipe.
    :param dataset: Source of the response measure and any randomization provenance.
    :param fold: Prepared fold whose mask checksums and pair order are recorded.
    :param mode: Canonical split mode.
    :param hpo: Resolved tuning settings.
    :param seed: Root seed of the run.
    :returns: Snapshot of fold metadata and run provenance.
    """
    metadata = copy.deepcopy(fold.metadata)
    metadata["measure"] = dataset.measure
    if "split_mode" in metadata:
        metadata["split_mode"] = mode
    if dataset.randomization_provenance is not None:
        metadata["randomization_provenance"] = copy.deepcopy(dataset.randomization_provenance)
    metadata["execution"] = {
        "config": model.config.model_dump(mode="json"),
        "scopes": _fold_provenance(fold),
        "hpo": hpo,
        "seed": seed,
    }
    return metadata

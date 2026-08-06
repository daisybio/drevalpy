"""Normalize predictor recipe strings and mappings into canonical config fields."""

from __future__ import annotations

from typing import Any

from drevalpy.components.registry import get_predictor
from drevalpy.models.config._space_defaults import split_space_and_options

_RESERVED_PREDICTOR_KEYS = frozenset({"name", "hyperparameters", "hyperparameter_space"})


def _reject_predictor_options(name: str, options: dict[str, Any]) -> None:
    """Reject values that the predictor does not declare as tunable.

    A template records a search space, not concrete constructor arguments, so a value that
    matches no declared hyperparameter has nowhere to go.

    :param name: Predictor registry name.
    :param options: Values that matched no entry in the declared space.
    :raises ValueError: If *options* is non-empty.
    """
    if not options:
        return
    option_keys = ", ".join(sorted(repr(key) for key in options))
    msg = f"Predictor {name!r} template configs do not accept non-tunable options ({option_keys})."
    raise ValueError(msg)


def _normalize_one_key_predictor_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize the ``{"randomForest": {"n_estimators": 10}}`` notation.

    Loose values move the ``default`` of the matching entry in the predictor's declared
    space. An explicitly declared ``hyperparameter_space`` wins over the derived one.

    :param data: Single-entry mapping of predictor name to arguments.
    :returns: Mapping of canonical ``PredictorConfig`` fields.
    :raises ValueError: If the arguments are neither ``None`` nor a mapping.
    """
    name, body = next(iter(data.items()))
    if body is None:
        return {"name": str(name)}
    if not isinstance(body, dict):
        msg = f"Predictor {name!r} arguments must be a mapping when provided"
        raise ValueError(msg)
    payload = dict(body)
    hyperparameter_space = payload.pop("hyperparameter_space", None)
    if payload:
        derived_space, options = split_space_and_options(get_predictor(str(name)), payload)
        _reject_predictor_options(str(name), options)
        hyperparameter_space = {**derived_space, **(hyperparameter_space or {})}
    result: dict[str, Any] = {"name": str(name)}
    if hyperparameter_space is not None:
        result["hyperparameter_space"] = hyperparameter_space
    return result


def normalize_predictor_config(data: Any) -> dict[str, Any]:
    """Normalize any accepted predictor notation into a canonical field mapping.

    Accepts a bare name (``"elasticNet"``), a one-key mapping carrying loose parameter
    values, or a mapping that already names its predictor, which normalizes to itself.

    :param data: Recipe string or field mapping.
    :returns: Mapping of canonical ``PredictorConfig`` fields.
    :raises ValueError: If a mapping is neither a one-key shorthand nor has ``name``.
    :raises TypeError: If *data* is not a string or mapping.
    """
    if isinstance(data, str):
        return {"name": data}

    if not isinstance(data, dict):
        msg = f"Predictor config must be a string or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        return dict(data)

    if not _RESERVED_PREDICTOR_KEYS.intersection(data.keys()) and len(data) == 1:
        return _normalize_one_key_predictor_dict(data)

    msg = "Predictor config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)

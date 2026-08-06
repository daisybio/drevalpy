"""Normalize predictor recipe strings and mappings into canonical config fields."""

from __future__ import annotations

from typing import Any

_RESERVED_PREDICTOR_KEYS = frozenset({"name", "hyperparameters", "hyperparameter_space"})


def _reject_predictor_options(name: str, options: dict[str, Any], *, context: str) -> None:
    if not options:
        return
    option_keys = ", ".join(sorted(repr(key) for key in options))
    msg = f"Predictor {name!r} {context} ({option_keys})."
    raise ValueError(msg)


def _merge_predictor_hyperparameter_space(
    existing: dict[str, Any] | None,
    derived: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if derived is None:
        return existing
    if existing is None:
        return derived
    return {**derived, **existing}


def _lift_named_predictor_hyperparameters(normalized: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = normalized.pop("hyperparameters", None)
    if not hyperparameters:
        return normalized
    if not isinstance(hyperparameters, dict):
        msg = "Predictor hyperparameters must be a mapping when provided"
        raise ValueError(msg)
    name = str(normalized["name"])
    space, options = _predictor_space_and_options(name, hyperparameters)
    if space is not None:
        normalized["hyperparameter_space"] = _merge_predictor_hyperparameter_space(
            normalized.get("hyperparameter_space"),
            space,
        )
    _reject_predictor_options(
        name,
        options,
        context="template configs no longer store non-tunable constructor options",
    )
    return normalized


def _normalize_named_predictor_dict(data: dict[str, Any]) -> dict[str, Any]:
    return _lift_named_predictor_hyperparameters(dict(data))


def _normalize_one_key_predictor_dict(data: dict[str, Any]) -> dict[str, Any]:
    name, body = next(iter(data.items()))
    if body is None:
        return {"name": str(name)}
    if not isinstance(body, dict):
        msg = f"Predictor {name!r} arguments must be a mapping when provided"
        raise ValueError(msg)
    payload = dict(body)
    hyperparameter_space = payload.pop("hyperparameter_space", None)
    if payload:
        derived_space, options = _predictor_space_and_options(str(name), payload)
        _reject_predictor_options(
            str(name),
            options,
            context="template shorthand no longer accepts non-tunable options",
        )
        hyperparameter_space = _merge_predictor_hyperparameter_space(hyperparameter_space, derived_space)
    result: dict[str, Any] = {"name": str(name)}
    if hyperparameter_space is not None:
        result["hyperparameter_space"] = hyperparameter_space
    return result


def normalize_predictor_config(data: Any) -> dict[str, Any]:
    """Normalize string or one-key mapping predictor configs.

    :param data: data.
    :returns: Result.
    :raises ValueError: Raised on invalid input.
    :raises TypeError: Raised on invalid input.
    """
    if isinstance(data, str):
        return {"name": data}

    if not isinstance(data, dict):
        msg = f"Predictor config must be a string or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        return _normalize_named_predictor_dict(data)

    if not _RESERVED_PREDICTOR_KEYS.intersection(data.keys()) and len(data) == 1:
        return _normalize_one_key_predictor_dict(data)

    msg = "Predictor config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)


def _predictor_space_and_options(name: str, values: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from drevalpy.components.registry import get_predictor

    cls = get_predictor(name)
    space = {
        key: dict(spec) if isinstance(spec, dict) else spec for key, spec in cls.get_hyperparameter_space().items()
    }
    options: dict[str, Any] = {}
    for key, value in values.items():
        if key in space and isinstance(space[key], dict):
            space[key] = {**space[key], "default": value}
        else:
            options[key] = value
    return (space if values else None), options

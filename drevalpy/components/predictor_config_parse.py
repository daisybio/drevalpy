"""Parse compact predictor config shorthand into normalized dicts."""

from __future__ import annotations

from typing import Any

_RESERVED_PREDICTOR_KEYS = frozenset({"name", "hyperparameters", "hyperparameter_space"})


def normalize_predictor_config(data: Any) -> dict[str, Any]:
    """Normalize string or one-key mapping predictor configs."""
    if isinstance(data, str):
        return {"name": data, "hyperparameters": {}}

    if not isinstance(data, dict):
        msg = f"Predictor config must be a string or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        normalized = dict(data)
        normalized["hyperparameters"] = dict(normalized.get("hyperparameters") or {})
        return normalized

    if not _RESERVED_PREDICTOR_KEYS.intersection(data.keys()) and len(data) == 1:
        name, hyperparameters = next(iter(data.items()))
        if hyperparameters is None:
            payload: dict[str, Any] = {}
        elif isinstance(hyperparameters, dict):
            payload = dict(hyperparameters)
        else:
            msg = f"Predictor {name!r} arguments must be a mapping when provided"
            raise ValueError(msg)
        return {"name": str(name), "hyperparameters": payload}

    msg = "Predictor config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)

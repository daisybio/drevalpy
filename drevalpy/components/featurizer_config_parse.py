"""Parse compact featurizer config shorthand into normalized dicts."""

from __future__ import annotations

from typing import Any

_RESERVED_FEATURIZER_KEYS = frozenset(
    {"name", "hyperparameters", "registry", "view", "views", "hyperparameter_space"}
)


def normalize_featurizer_config(data: Any, *, default_registry: str = "cell_line") -> dict[str, Any]:
    """Normalize string or one-key mapping featurizer configs."""
    if isinstance(data, str):
        return {"name": data, "hyperparameters": {}, "registry": default_registry}

    if not isinstance(data, dict):
        msg = f"Featurizer config must be a string or mapping, got {type(data)!r}"
        raise TypeError(msg)

    if "name" in data:
        normalized = dict(data)
        normalized.setdefault("registry", default_registry)
        hyperparameters = dict(normalized.get("hyperparameters") or {})
        if "featurizers" in hyperparameters:
            registry = str(normalized.get("registry", default_registry))
            hyperparameters["featurizers"] = [
                normalize_featurizer_config(item, default_registry=registry) for item in hyperparameters["featurizers"]
            ]
        normalized["hyperparameters"] = hyperparameters
        return normalized

    if not _RESERVED_FEATURIZER_KEYS.intersection(data.keys()) and len(data) == 1:
        name, hyperparameters = next(iter(data.items()))
        if hyperparameters is None:
            payload: dict[str, Any] = {}
        elif isinstance(hyperparameters, dict):
            payload = dict(hyperparameters)
        else:
            msg = f"Featurizer {name!r} arguments must be a mapping when provided"
            raise ValueError(msg)
        if "featurizers" in payload:
            payload["featurizers"] = [
                normalize_featurizer_config(item, default_registry=default_registry) for item in payload["featurizers"]
            ]
        return {"name": str(name), "hyperparameters": payload, "registry": default_registry}

    msg = "Featurizer config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)

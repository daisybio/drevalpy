"""Parse compact featurizer config shorthand into normalized dicts."""

from __future__ import annotations

from typing import Any

_RESERVED_FEATURIZER_KEYS = frozenset({"name", "hyperparameters", "registry", "view", "views", "hyperparameter_space"})
_CONCAT_FEATURIZER_NAME = "concatFeaturizers"


def _parse_featurizer_token(token: str, *, default_registry: str) -> dict[str, Any]:
    """Normalize a bare featurizer token, including ``+`` concat recipes."""
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    if "+" not in trimmed:
        return {"name": trimmed, "hyperparameters": {}, "registry": default_registry}

    parts = [part.strip() for part in trimmed.split("+")]
    if any(not part for part in parts):
        msg = "Featurizer recipe segments joined by '+' must be non-empty"
        raise ValueError(msg)
    if len(parts) == 1:
        return {"name": parts[0], "hyperparameters": {}, "registry": default_registry}

    return {
        "name": _CONCAT_FEATURIZER_NAME,
        "hyperparameters": {
            "featurizers": [normalize_featurizer_config(part, default_registry=default_registry) for part in parts],
        },
        "registry": default_registry,
    }


def normalize_featurizer_config(data: Any, *, default_registry: str = "cell_line") -> dict[str, Any]:
    """Normalize string, list, or one-key mapping featurizer configs."""
    if isinstance(data, str):
        return _parse_featurizer_token(data, default_registry=default_registry)

    if isinstance(data, list):
        if not data:
            msg = "Featurizer list must be non-empty"
            raise ValueError(msg)
        return {
            "name": _CONCAT_FEATURIZER_NAME,
            "hyperparameters": {
                "featurizers": [normalize_featurizer_config(item, default_registry=default_registry) for item in data],
            },
            "registry": default_registry,
        }

    if not isinstance(data, dict):
        msg = f"Featurizer config must be a string, list, or mapping, got {type(data)!r}"
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

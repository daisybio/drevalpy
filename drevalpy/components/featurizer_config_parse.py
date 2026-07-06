"""Parse compact featurizer config shorthand into normalized dicts."""

from __future__ import annotations

import re
from typing import Any

from drevalpy.components.featurizer_label import requires_explicit_view
from drevalpy.components.view_aliases import resolve_omics_view

_RESERVED_FEATURIZER_KEYS = frozenset({"name", "hyperparameters", "registry", "view", "views", "hyperparameter_space"})
_CONCAT_FEATURIZER_NAME = "concatFeaturizers"
_BRACKET_ATOM_RE = re.compile(r"^([^[\]]+)\[([^\]]+)\]$")
_VIEW_PARAMETRIC_FEATURIZERS = frozenset({"raw", "pca"})


def _split_concat_recipe(token: str) -> list[str]:
    """Split a concat recipe on ``+`` outside square brackets."""
    if token.startswith("+") or token.endswith("+") or "++" in token:
        msg = "Featurizer recipe segments joined by '+' must be non-empty"
        raise ValueError(msg)
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in token:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "+" and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _parse_bracket_atom_name(name_token: str, *, default_registry: str) -> tuple[str, str | None]:
    """Parse ``name[view]`` into registry name and resolved view."""
    match = _BRACKET_ATOM_RE.match(name_token.strip())
    if not match:
        return name_token.strip(), None
    name, view_token = match.groups()
    name = name.strip()
    if name not in _VIEW_PARAMETRIC_FEATURIZERS:
        msg = f"Bracket syntax is only supported for raw and pca, got {name!r}"
        raise ValueError(msg)
    if default_registry != "cell_line":
        msg = f"Bracket view syntax is only supported for cell-line featurizers, got registry {default_registry!r}"
        raise ValueError(msg)
    return name, resolve_omics_view(view_token)


def _validate_view_required(config: dict[str, Any]) -> None:
    name = str(config.get("name", ""))
    if not requires_explicit_view(name):
        return
    view = config.get("view")
    if view is None or (isinstance(view, str) and not view.strip()):
        msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
        raise ValueError(msg)


def _finalize_featurizer_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    _validate_view_required(normalized)
    return normalized


def _parse_featurizer_atom(token: str, *, default_registry: str) -> dict[str, Any]:
    """Normalize one featurizer atom, including optional ``name[view]`` syntax."""
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    name, view = _parse_bracket_atom_name(trimmed, default_registry=default_registry)
    if view is None and name in _VIEW_PARAMETRIC_FEATURIZERS:
        msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
        raise ValueError(msg)
    payload: dict[str, Any] = {"name": name, "hyperparameters": {}, "registry": default_registry}
    if view is not None:
        payload["view"] = view
    return _finalize_featurizer_config(payload)


def _parse_featurizer_token(token: str, *, default_registry: str) -> dict[str, Any]:
    """Normalize a bare featurizer token, including ``+`` concat recipes."""
    trimmed = token.strip()
    if not trimmed:
        msg = "Featurizer token must be a non-empty string"
        raise ValueError(msg)
    parts = _split_concat_recipe(trimmed)
    if len(parts) == 1:
        return _parse_featurizer_atom(parts[0], default_registry=default_registry)

    if any(not part for part in parts):
        msg = "Featurizer recipe segments joined by '+' must be non-empty"
        raise ValueError(msg)

    return {
        "name": _CONCAT_FEATURIZER_NAME,
        "hyperparameters": {
            "featurizers": [_parse_featurizer_atom(part, default_registry=default_registry) for part in parts],
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
        return _finalize_featurizer_config(normalized)

    if not _RESERVED_FEATURIZER_KEYS.intersection(data.keys()) and len(data) == 1:
        name_token, hyperparameters = next(iter(data.items()))
        if hyperparameters is None:
            payload: dict[str, Any] = {}
        elif isinstance(hyperparameters, dict):
            payload = dict(hyperparameters)
        else:
            msg = f"Featurizer {name_token!r} arguments must be a mapping when provided"
            raise ValueError(msg)
        name, view = _parse_bracket_atom_name(str(name_token), default_registry=default_registry)
        if view is None and name in _VIEW_PARAMETRIC_FEATURIZERS:
            msg = f"Featurizer {name!r} requires an explicit view, e.g. {name}[expression]"
            raise ValueError(msg)
        config: dict[str, Any] = {"name": name, "hyperparameters": payload, "registry": default_registry}
        if view is not None:
            config["view"] = view
        if "featurizers" in payload:
            payload["featurizers"] = [
                normalize_featurizer_config(item, default_registry=default_registry) for item in payload["featurizers"]
            ]
        return _finalize_featurizer_config(config)

    msg = "Featurizer config must be a string, one-key mapping, or dict with 'name'"
    raise ValueError(msg)

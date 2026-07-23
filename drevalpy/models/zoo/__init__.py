"""Built-in zoo entries for passing official drevalpy models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.models.config import ModelConfig
from drevalpy.models.config_io import model_config_from_dict, model_config_from_yaml
from drevalpy.models.flat_hyperparameters import apply_public_flat_hyperparameters
from drevalpy.types.model_scope import ModelScope

_BUILTIN_ZOO_DIR = Path(__file__).resolve().parent
_EXTERNAL_ZOO: dict[str, ModelConfig] = {}
_VALIDATED_BUILTIN_NAMES: set[str] = set()


def _load_builtin_entries() -> dict[str, ModelConfig]:
    entries: dict[str, ModelConfig] = {}
    for path in sorted(_BUILTIN_ZOO_DIR.glob("*.yaml")):
        config = model_config_from_yaml(path)
        entries[path.stem] = config
    return entries


_BUILTIN_ZOO = _load_builtin_entries()
_BUILTIN_ZOO_NAMES = frozenset(_BUILTIN_ZOO)


def _coerce_scope(scope: ModelScope | str | None) -> ModelScope | None:
    if scope is None:
        return None
    if isinstance(scope, ModelScope):
        return scope
    return ModelScope(scope)


def list_zoo_names(
    *,
    include_external: bool = True,
    scope: ModelScope | str | None = None,
) -> list[str]:
    """Return sorted built-in (and optional external) zoo entry names.

    :param include_external: Include externally registered zoo entries.
    :param scope: Optional ``ModelScope`` (or its string value) filter for
        multi-drug vs single-drug presets.
    """
    names = set(_BUILTIN_ZOO)
    if include_external:
        names.update(_EXTERNAL_ZOO)
    resolved_scope = _coerce_scope(scope)
    if resolved_scope is None:
        return sorted(names)
    filtered = [name for name in names if get_zoo_config(name).scope == resolved_scope]
    return sorted(filtered)


def get_zoo_config(name: str) -> ModelConfig:
    """Return a copy of a zoo entry by name."""
    if name in _EXTERNAL_ZOO:
        return _clone_model_config(_EXTERNAL_ZOO[name])
    if name not in _BUILTIN_ZOO:
        msg = f"Unknown zoo entry: {name}"
        raise KeyError(msg)
    if name not in _VALIDATED_BUILTIN_NAMES:
        _BUILTIN_ZOO[name].validate()
        _VALIDATED_BUILTIN_NAMES.add(name)
    return _clone_model_config(_BUILTIN_ZOO[name])


def register_external_zoo_entry(name: str, config: ModelConfig, *, replace: bool = True) -> None:
    """Register an external zoo entry.

    External entries are resolved through ``ModelConfig`` / ``construct_model``
    rather than dynamically extending an already-built ``MODEL_FACTORY``.
    Built-in names are rejected by default.
    """
    if name in _BUILTIN_ZOO_NAMES:
        msg = f"External zoo entry {name!r} collides with a built-in preset"
        raise ValueError(msg)
    if name in _EXTERNAL_ZOO and not replace:
        msg = f"External zoo entry {name!r} is already registered"
        raise ValueError(msg)
    config.validate()
    _EXTERNAL_ZOO[name] = _clone_model_config(config)


def clear_external_zoo() -> None:
    """Remove all externally registered zoo entries (primarily for tests)."""
    _EXTERNAL_ZOO.clear()


def load_external_zoo_file(path: Path | str) -> list[str]:
    """Load one or more zoo entries from a YAML file.

    Validates the complete file before mutating global external zoo state.
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"External zoo YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"External zoo YAML must contain a mapping: {yaml_path}"
        raise ValueError(msg)

    parsed: list[tuple[str, ModelConfig]] = []
    if "predictor" in data:
        payload = dict(data)
        entry_name = str(payload.pop("name", yaml_path.stem))
        try:
            config = model_config_from_dict(payload, source=yaml_path)
            config.validate()
        except ValueError as exc:
            msg = f"Invalid zoo entry {entry_name!r} in {yaml_path}: {exc}"
            raise ValueError(msg) from exc
        if entry_name in _BUILTIN_ZOO_NAMES:
            msg = f"External zoo entry {entry_name!r} collides with a built-in preset"
            raise ValueError(msg)
        parsed.append((entry_name, config))
    else:
        for entry_name, entry_data in data.items():
            if not isinstance(entry_data, dict):
                msg = f"Zoo entry '{entry_name}' must be a mapping in {yaml_path}"
                raise ValueError(msg)
            payload = dict(entry_data)
            payload.pop("name", None)
            name = str(entry_name)
            try:
                config = model_config_from_dict(payload, source=yaml_path)
                config.validate()
            except ValueError as exc:
                msg = f"Invalid zoo entry {name!r} in {yaml_path}: {exc}"
                raise ValueError(msg) from exc
            if name in _BUILTIN_ZOO_NAMES:
                msg = f"External zoo entry {name!r} collides with a built-in preset"
                raise ValueError(msg)
            parsed.append((name, config))

    for entry_name, config in parsed:
        _EXTERNAL_ZOO[entry_name] = _clone_model_config(config)
    return [entry_name for entry_name, _ in parsed]


def zoo_model_config(name: str, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
    """Return a zoo config with optional public flat hyperparameter overrides."""
    config = get_zoo_config(name)
    if not hyperparameters:
        return config
    return apply_public_flat_hyperparameters(config, hyperparameters)


def _clone_model_config(config: ModelConfig) -> ModelConfig:
    return config.model_copy(deep=True)

"""Built-in zoo entries for passing official drevalpy models."""

from __future__ import annotations

from typing import Any

from upath import UPath as Path

from drevalpy.models.config import ModelConfig, ResolvedModelConfig
from drevalpy.models.config.io import from_yaml
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

from ._external_load import (
    _collect_zoo_entries_from_yaml,
    _load_zoo_yaml_mapping,
)

_BUILTIN_ZOO_DIR = Path(__file__).resolve().parent
_EXTERNAL_ZOO: dict[str, ModelConfig] = {}


def _load_builtin_entries() -> dict[str, ModelConfig]:
    entries: dict[str, ModelConfig] = {}
    for path in sorted(_BUILTIN_ZOO_DIR.glob("*.yaml")):
        config = from_yaml(path)
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
    :param scope: Optional ``ModelScope`` (or its string value) filter for multi-drug vs single-drug presets.
    :returns: Sorted list of zoo preset names matching the filters.
    """
    names = set(_BUILTIN_ZOO)
    if include_external:
        names.update(_EXTERNAL_ZOO)
    resolved_scope = _coerce_scope(scope)
    if resolved_scope is None:
        return sorted(names)
    filtered = [name for name in names if get_zoo_config(name).scope == resolved_scope]
    return sorted(filtered)


def get_zoo_config(name: str, *, prediction_mode: PredictionMode | None = None) -> ModelConfig:
    """Return a copy of a zoo entry by name.

    :param name: Built-in or externally registered zoo preset name.
    :param prediction_mode: Optional prediction mode overriding the preset's own value.
    :returns: Deep copy of the zoo ``ModelConfig``.
    :raises KeyError: If ``name`` is not a known zoo entry.
    """
    if name in _EXTERNAL_ZOO:
        return _clone_model_config(_EXTERNAL_ZOO[name], prediction_mode=prediction_mode)
    if name not in _BUILTIN_ZOO:
        msg = f"Unknown zoo entry: {name}"
        raise KeyError(msg)
    return _clone_model_config(_BUILTIN_ZOO[name], prediction_mode=prediction_mode)


def register_external_zoo_entry(name: str, config: ModelConfig, *, replace: bool = True) -> None:
    """Register an external zoo entry.

    External entries are resolved through ``ModelConfig`` / ``construct_model``
    rather than dynamically extending an already-built ``MODEL_FACTORY``.

    :param name: Unique preset name; must not collide with built-in names unless ``replace`` allows replacement.
    :param config: Validated model configuration for the preset.
    :param replace: Allow replacing an existing external entry with the same name.
    :raises ValueError: If ``name`` collides with a built-in preset.
    """
    if name in _BUILTIN_ZOO_NAMES:
        msg = f"External zoo entry {name!r} collides with a built-in preset"
        raise ValueError(msg)
    if name in _EXTERNAL_ZOO and not replace:
        msg = f"External zoo entry {name!r} is already registered"
        raise ValueError(msg)
    _EXTERNAL_ZOO[name] = _clone_model_config(config)


def clear_external_zoo() -> None:
    """Remove all externally registered zoo entries (primarily for tests)."""
    _EXTERNAL_ZOO.clear()


def load_external_zoo_file(path: Path | str) -> list[str]:
    """Load one or more zoo entries from a YAML file.

    Validates the complete file before mutating global external zoo state.

    :param path: YAML file mapping preset names to model configs.
    :returns: Names of entries loaded from the file.
    """
    yaml_path = Path(path)
    data = _load_zoo_yaml_mapping(yaml_path)
    parsed = _collect_zoo_entries_from_yaml(data, source=yaml_path, builtin_names=_BUILTIN_ZOO_NAMES)
    for entry_name, config in parsed:
        _EXTERNAL_ZOO[entry_name] = _clone_model_config(config)
    return [entry_name for entry_name, _ in parsed]


def zoo_model_config(
    name: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    prediction_mode: PredictionMode | None = None,
) -> ModelConfig | ResolvedModelConfig:
    """Return a zoo config with optional public flat hyperparameter overrides.

    When *hyperparameters* are provided, returns a ``ResolvedModelConfig``.

    :param name: Built-in or external zoo preset name.
    :param hyperparameters: Optional flat public overrides applied to the preset.
    :param prediction_mode: Optional prediction mode overriding the preset's own value.
    :returns: ``ModelConfig`` copy, or resolved config when overrides are provided.
    """
    config = get_zoo_config(name, prediction_mode=prediction_mode)
    if not hyperparameters:
        return config
    from drevalpy.models.tuning.public_flat import apply_public_hyperparameters_to_config

    return apply_public_hyperparameters_to_config(config, hyperparameters)


def _clone_model_config(config: ModelConfig, *, prediction_mode: PredictionMode | None = None) -> ModelConfig:
    payload = config.model_dump(mode="python")
    if prediction_mode is not None:
        payload["prediction_mode"] = prediction_mode
    return ModelConfig.model_validate(payload)

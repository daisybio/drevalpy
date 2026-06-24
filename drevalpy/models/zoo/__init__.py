"""Built-in zoo entries for passing official drevalpy models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.components.config import ModelConfig
from drevalpy.models.config_io import model_config_from_dict, model_config_from_yaml

_BUILTIN_ZOO_DIR = Path(__file__).resolve().parent
_EXTERNAL_ZOO: dict[str, ModelConfig] = {}


def _load_builtin_entries() -> dict[str, ModelConfig]:
    entries: dict[str, ModelConfig] = {}
    for path in sorted(_BUILTIN_ZOO_DIR.glob("*.yaml")):
        config = model_config_from_yaml(path)
        entries[path.stem] = config
    return entries


_BUILTIN_ZOO = _load_builtin_entries()


def list_zoo_names(*, include_external: bool = True) -> list[str]:
    """Return sorted built-in (and optional external) zoo entry names."""
    names = set(_BUILTIN_ZOO)
    if include_external:
        names.update(_EXTERNAL_ZOO)
    return sorted(names)


def get_zoo_config(name: str) -> ModelConfig:
    """Return a copy of a zoo entry by name."""
    if name in _EXTERNAL_ZOO:
        return _clone_model_config(_EXTERNAL_ZOO[name])
    if name not in _BUILTIN_ZOO:
        msg = f"Unknown zoo entry: {name}"
        raise KeyError(msg)
    return _clone_model_config(_BUILTIN_ZOO[name])


def register_external_zoo_entry(name: str, config: ModelConfig) -> None:
    """Register or replace an external zoo entry."""
    _EXTERNAL_ZOO[name] = _clone_model_config(config)


def load_external_zoo_file(path: Path | str) -> list[str]:
    """Load one or more zoo entries from a YAML file."""
    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"External zoo YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"External zoo YAML must contain a mapping: {yaml_path}"
        raise ValueError(msg)

    loaded: list[str] = []
    if "predictor" in data:
        payload = dict(data)
        entry_name = str(payload.pop("name", yaml_path.stem))
        try:
            register_external_zoo_entry(entry_name, model_config_from_dict(payload, source=yaml_path))
        except ValueError as exc:
            msg = f"Invalid zoo entry {entry_name!r} in {yaml_path}: {exc}"
            raise ValueError(msg) from exc
        loaded.append(entry_name)
        return loaded

    for entry_name, entry_data in data.items():
        if not isinstance(entry_data, dict):
            msg = f"Zoo entry '{entry_name}' must be a mapping in {yaml_path}"
            raise ValueError(msg)
        payload = dict(entry_data)
        payload.pop("name", None)
        try:
            register_external_zoo_entry(str(entry_name), model_config_from_dict(payload, source=yaml_path))
        except ValueError as exc:
            msg = f"Invalid zoo entry {entry_name!r} in {yaml_path}: {exc}"
            raise ValueError(msg) from exc
        loaded.append(str(entry_name))
    return loaded


def zoo_model_config(name: str, hyperparameters: dict[str, Any] | None = None) -> ModelConfig:
    """Return a zoo config with optional predictor and view hyperparameter overrides."""
    from drevalpy.models.factory import featurizer_configs_from_view_hyperparameters

    config = get_zoo_config(name)
    if not hyperparameters:
        return config
    merged_hp = {**config.predictor.hyperparameters, **hyperparameters}
    cell_line_featurizer = config.cell_line_featurizer
    drug_featurizer = config.drug_featurizer
    cell_line_override, drug_override = featurizer_configs_from_view_hyperparameters(hyperparameters)
    if cell_line_override is not None:
        cell_line_featurizer = cell_line_override
    if drug_override is not None:
        drug_featurizer = drug_override
    return config.model_copy(
        update={
            "cell_line_featurizer": cell_line_featurizer,
            "drug_featurizer": drug_featurizer,
            "predictor": config.predictor.model_copy(update={"hyperparameters": merged_hp}),
        },
        deep=True,
    )


def _clone_model_config(config: ModelConfig) -> ModelConfig:
    return config.model_copy(deep=True)

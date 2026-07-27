"""Load external zoo YAML entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.models.config import ModelConfig
from drevalpy.models.config_io import model_config_from_dict


def _load_zoo_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"External zoo YAML not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"External zoo YAML must contain a mapping: {path}"
        raise ValueError(msg)
    return data


def _assert_not_builtin_zoo_name(entry_name: str, builtin_names: frozenset[str]) -> None:
    if entry_name in builtin_names:
        msg = f"External zoo entry {entry_name!r} collides with a built-in preset"
        raise ValueError(msg)


def _parse_zoo_entry(
    entry_name: str,
    payload: dict[str, Any],
    *,
    source: Path,
    builtin_names: frozenset[str],
) -> tuple[str, ModelConfig]:
    _assert_not_builtin_zoo_name(entry_name, builtin_names)
    try:
        config = model_config_from_dict(payload, source=source)
        config.validate()
    except ValueError as exc:
        msg = f"Invalid zoo entry {entry_name!r} in {source}: {exc}"
        raise ValueError(msg) from exc
    return entry_name, config


def _collect_zoo_entries_from_yaml(
    data: dict[str, Any],
    *,
    source: Path,
    builtin_names: frozenset[str],
) -> list[tuple[str, ModelConfig]]:
    parsed: list[tuple[str, ModelConfig]] = []
    if "predictor" in data:
        payload = dict(data)
        entry_name = str(payload.pop("name", source.stem))
        parsed.append(_parse_zoo_entry(entry_name, payload, source=source, builtin_names=builtin_names))
        return parsed

    for entry_name, entry_data in data.items():
        if not isinstance(entry_data, dict):
            msg = f"Zoo entry '{entry_name}' must be a mapping in {source}"
            raise ValueError(msg)
        payload = dict(entry_data)
        payload.pop("name", None)
        name = str(entry_name)
        parsed.append(_parse_zoo_entry(name, payload, source=source, builtin_names=builtin_names))
    return parsed

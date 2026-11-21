"""The model zoo: named model configurations, built in and external.

Every preset lives in one :class:`~drevalpy.registry._registry.Registry` of ``ModelConfig``,
reached through :func:`zoo_registry`. The built-in presets are the ``*.yaml`` files in this
directory, parsed on first access; each carries ``builtin=True`` in its registry metadata.
External presets come from :func:`register_external_zoo_entry` and
:func:`load_external_zoo_file`. An external preset may replace an external one of the same
name, never a built-in one.
"""

from __future__ import annotations

from typing import Any

from upath import UPath as Path

from drevalpy.log import get_logger
from drevalpy.models.config import ModelConfig
from drevalpy.models.config.io import from_dict, from_yaml, read_yaml_mapping
from drevalpy.registry._registry import Registry, Snapshot
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

_BUILTIN_ZOO_DIR = Path(__file__).resolve().parent
logger = get_logger(__name__)
_ZOO: Registry[ModelConfig] = Registry("zoo entry")


def _load_builtin_entries() -> dict[str, ModelConfig]:
    """Parse every ``*.yaml`` preset shipped in this directory.

    A preset that fails validation is logged and skipped, so one broken file cannot hide
    the others.

    :returns: Mapping of preset name (the file stem) to its model config.
    """
    entries: dict[str, ModelConfig] = {}
    for path in sorted(_BUILTIN_ZOO_DIR.glob("*.yaml")):
        try:
            config = from_yaml(path)
        except ValueError as exc:
            # tests/models/test_model_roundtrip.py fails on a skipped preset; users only get the warning.
            logger.warning("Skipping built-in zoo preset %s, which fails validation: %s", path.name, exc)
            continue
        entries[path.stem] = config
    return entries


def zoo_registry() -> Registry[ModelConfig]:
    """Return the zoo registry, parsing the built-in presets on first access.

    The built-ins are parsed again whenever the registry is empty, which is also how a
    test's registry snapshot taken before the first access restores cleanly.

    :returns: The registry of every zoo preset, built-in ones first.
    """
    if not _ZOO:
        for name, config in _load_builtin_entries().items():
            _ZOO.register(name, config, metadata={"builtin": True})
    return _ZOO


def _builtin_names() -> frozenset[str]:
    registry = zoo_registry()
    return frozenset(name for name in registry.names() if registry.metadata(name).get("builtin"))


def list_zoo_names(
    *,
    include_external: bool = True,
    scope: ModelScope | str | None = None,
) -> list[str]:
    """Return sorted zoo preset names, optionally only the built-in ones or one scope.

    :param include_external: Include externally registered zoo entries.
    :param scope: Optional ``ModelScope`` (or its string value) filter for multi-drug vs single-drug presets.
    :returns: Sorted list of zoo preset names matching the filters.
    """
    registry = zoo_registry()
    builtin = _builtin_names()
    wanted = None if scope is None else ModelScope(scope)
    return sorted(
        name
        for name, config in registry.items()
        if (include_external or name in builtin) and (wanted is None or config.scope == wanted)
    )


def get_zoo_config(name: str, *, prediction_mode: PredictionMode | None = None) -> ModelConfig:
    """Return a zoo entry by name.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`. Configs are deep-frozen,
    so the entry itself is returned and shared safely.

    :param name: Built-in or externally registered zoo preset name.
    :param prediction_mode: Optional prediction mode overriding the preset's own; the config is
        validated again with it, since the predictor must support the mode.
    :returns: The zoo ``ModelConfig``.
    """
    config = zoo_registry().get(name)
    if prediction_mode is None:
        return config
    return ModelConfig.model_validate({**config.model_dump(mode="python"), "prediction_mode": prediction_mode})


def _assert_not_builtin(name: str) -> None:
    if name in _builtin_names():
        raise ValueError(f"External zoo entry {name!r} collides with a built-in preset")


def register_external_zoo_entry(name: str, config: ModelConfig, *, replace: bool = True) -> None:
    """Register an external zoo entry.

    Raises ``ValueError`` if ``name`` collides with a built-in preset, or is already
    registered and *replace* is false.

    :param name: Preset name; must not be a built-in preset name.
    :param config: Validated model configuration for the preset.
    :param replace: Allow replacing an existing external entry with the same name.
    """
    _assert_not_builtin(name)
    zoo_registry().register(name, config, replace=replace)


def clear_external_zoo() -> None:
    """Remove all externally registered zoo entries, keeping the built-in ones."""
    registry = zoo_registry()
    builtin = _builtin_names()
    kept = tuple(entry for entry in registry.snapshot().entries if entry[0] in builtin)
    registry.restore(Snapshot(registry.kind, kept))


def _parse_zoo_file(data: dict[str, Any], source: Path) -> list[tuple[str, ModelConfig]]:
    """Validate every entry of an external zoo file.

    A file holding one config (it has a ``predictor`` key) names its entry by its ``name``
    key, else by the file stem; any other file maps entry names to configs.

    :param data: Parsed YAML mapping.
    :param source: The file, for the messages and the default entry name.
    :returns: ``(name, config)`` per entry, in file order.
    :raises ValueError: If an entry is not a mapping, collides with a built-in preset or fails validation.
    """
    if "predictor" in data:
        payload = dict(data)
        entries = {str(payload.pop("name", source.stem)): payload}
    else:
        entries = {str(name): entry for name, entry in data.items()}
    parsed = []
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Zoo entry {name!r} must be a mapping in {source}")
        _assert_not_builtin(name)
        payload = {key: value for key, value in entry.items() if key != "name"}
        try:
            parsed.append((name, from_dict(payload, source=source)))
        except ValueError as exc:
            raise ValueError(f"Invalid zoo entry {name!r} in {source}: {exc}") from exc
    return parsed


def load_external_zoo_file(path: Path | str) -> list[str]:
    """Load one or more zoo entries from a YAML file, replacing external entries of the same name.

    Validates the complete file before mutating the zoo. ``read_yaml_mapping`` raises
    ``FileNotFoundError`` for a missing file and ``TypeError`` for one that holds no mapping.

    :param path: YAML file mapping preset names to model configs, or holding one model config.
    :returns: Names of entries loaded from the file.
    """
    yaml_path = Path(path)
    parsed = _parse_zoo_file(read_yaml_mapping(yaml_path, "External zoo YAML"), yaml_path)
    for name, config in parsed:
        zoo_registry().register(name, config, replace=True)
    return [name for name, _ in parsed]

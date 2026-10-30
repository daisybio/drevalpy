"""The model zoo: named model configurations, built in and external.

Every preset lives in one :class:`~drevalpy.registry._registry.Registry` of ``ModelConfig``,
reached through :func:`zoo_registry`. The built-in presets are the ``*.yaml`` files in this
directory, parsed on first access; each carries ``builtin=True`` in its registry metadata.
External presets come from :func:`register_external_zoo_entry` and
:func:`load_external_zoo_file`. An external preset may replace an external one of the same
name, never a built-in one.
"""

from __future__ import annotations

from upath import UPath as Path

from drevalpy.log import get_logger
from drevalpy.models.config import ModelConfig
from drevalpy.models.config.io import from_yaml
from drevalpy.registry._registry import Registry, Snapshot
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode

from ._external_load import (
    _collect_zoo_entries_from_yaml,
    _load_zoo_yaml_mapping,
)

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
    """Return a copy of a zoo entry by name.

    An unknown *name* raises :class:`~drevalpy.errors.UnknownNameError`.

    :param name: Built-in or externally registered zoo preset name.
    :param prediction_mode: Optional prediction mode overriding the preset's own value.
    :returns: Deep copy of the zoo ``ModelConfig``.
    """
    return _clone_model_config(zoo_registry().get(name), prediction_mode=prediction_mode)


def _assert_not_builtin(name: str) -> None:
    if name in _builtin_names():
        msg = f"External zoo entry {name!r} collides with a built-in preset"
        raise ValueError(msg)


def register_external_zoo_entry(name: str, config: ModelConfig, *, replace: bool = True) -> None:
    """Register an external zoo entry.

    Raises ``ValueError`` if ``name`` collides with a built-in preset, or is already
    registered and *replace* is false.

    :param name: Preset name; must not be a built-in preset name.
    :param config: Validated model configuration for the preset.
    :param replace: Allow replacing an existing external entry with the same name.
    """
    _assert_not_builtin(name)
    zoo_registry().register(name, _clone_model_config(config), replace=replace)


def clear_external_zoo() -> None:
    """Remove all externally registered zoo entries, keeping the built-in ones."""
    registry = zoo_registry()
    builtin = _builtin_names()
    kept = tuple(entry for entry in registry.snapshot().entries if entry[0] in builtin)
    registry.restore(Snapshot(registry.kind, kept))


def load_external_zoo_file(path: Path | str) -> list[str]:
    """Load one or more zoo entries from a YAML file, replacing external entries of the same name.

    Validates the complete file before mutating the zoo.

    :param path: YAML file mapping preset names to model configs.
    :returns: Names of entries loaded from the file.
    """
    yaml_path = Path(path)
    data = _load_zoo_yaml_mapping(yaml_path)
    parsed = _collect_zoo_entries_from_yaml(data, source=yaml_path, builtin_names=_builtin_names())
    for entry_name, config in parsed:
        zoo_registry().register(entry_name, _clone_model_config(config), replace=True)
    return [entry_name for entry_name, _ in parsed]


def _clone_model_config(config: ModelConfig, *, prediction_mode: PredictionMode | None = None) -> ModelConfig:
    payload = config.model_dump(mode="python")
    if prediction_mode is not None:
        payload["prediction_mode"] = prediction_mode
    return ModelConfig.model_validate(payload)

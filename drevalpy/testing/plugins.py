"""Conformance check that an installed plugin actually reached the registries.

A plugin is wired up through three separate mechanisms - a ``drevalpy.plugins``
entry point, a module import, and one registration decorator per component - and
a mistake in any of them makes the components silently absent. :func:`check_plugin`
walks all three in order and reports the first that breaks, so a plugin's CI can
assert "my components are installed and reachable" in one call.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType

from drevalpy.registry import (
    cell_line_featurizer,
    drug_featurizer,
    get_failed_plugins,
    predictor,
    splitter,
    visualization,
)

#: Entry-point group third-party plugins declare themselves under.
ENTRY_POINT_GROUP = "drevalpy.plugins"

#: The registries a plugin can contribute to, keyed by the name used in reports.
_REGISTRIES: Mapping[str, ModuleType] = {
    "cell_line_featurizer": cell_line_featurizer,
    "drug_featurizer": drug_featurizer,
    "predictor": predictor,
    "splitter": splitter,
    "visualization": visualization,
}


class PluginCheckError(AssertionError):
    """Raised when an installed plugin is not reachable through the registries.

    Subclasses ``AssertionError`` so a failure reads as a test failure rather
    than an error when raised from inside a test.
    """


@dataclass(frozen=True)
class PluginReport:
    """What one plugin contributed, as observed through the public registries."""

    name: str
    value: str
    module: str
    components: Mapping[str, tuple[str, ...]]

    @property
    def component_names(self) -> tuple[str, ...]:
        """Every registered name the plugin contributed, across all registries."""
        return tuple(name for names in self.components.values() for name in names)


def _declared_entry_point(name: str) -> importlib.metadata.EntryPoint:
    """Return the ``drevalpy.plugins`` entry point called *name*.

    Args:
        name: Entry-point name, which is the distribution's own choice and is
            usually its import package name.

    Returns:
        The declared entry point.

    Raises:
        PluginCheckError: If no installed distribution declares it.
    """
    declared = {ep.name: ep for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}
    entry_point = declared.get(name)
    if entry_point is None:
        msg = (
            f"No {ENTRY_POINT_GROUP} entry point named {name!r}. "
            f"Declared: {sorted(declared) or 'none'}. "
            "Check the [project.entry-points] table and reinstall the plugin."
        )
        raise PluginCheckError(msg)
    return entry_point


def _recorded_failure(name: str) -> str | None:
    """Return the traceback the plugin loader recorded for *name*, if any.

    Args:
        name: Entry-point name.

    Returns:
        The formatted traceback, or ``None`` when the plugin did not fail.
    """
    return get_failed_plugins().get(name)


def _load(entry_point: importlib.metadata.EntryPoint) -> str:
    """Import *entry_point* and return the module it resolves to.

    Args:
        entry_point: Declared plugin entry point.

    Returns:
        Dotted module name the entry point targets.

    Raises:
        PluginCheckError: If discovery recorded a failure, or the import raises.
    """
    recorded = _recorded_failure(entry_point.name)
    if recorded is not None:
        msg = f"Plugin {entry_point.name!r} failed to load during discovery:\n{recorded}"
        raise PluginCheckError(msg)
    try:
        entry_point.load()
    except Exception as exc:
        msg = f"Plugin {entry_point.name!r} declares {entry_point.value!r}, which failed to import: {exc!r}"
        raise PluginCheckError(msg) from exc
    return entry_point.module


def _contributed(root_package: str) -> dict[str, tuple[str, ...]]:
    """Return every registered name whose implementation lives in *root_package*.

    Resolution goes through each registry's public ``get``, so a name that is
    listed but cannot be retrieved surfaces here rather than at model build time.

    Args:
        root_package: Top-level import package of the plugin.

    Returns:
        Mapping of registry name to the sorted names it holds for the plugin.
    """
    contributed: dict[str, tuple[str, ...]] = {}
    for registry_name, module in _REGISTRIES.items():
        owned = [
            name for name in module.list() if getattr(module.get(name), "__module__", "").split(".")[0] == root_package
        ]
        if owned:
            contributed[registry_name] = tuple(sorted(owned))
    return contributed


def check_plugin(name: str) -> PluginReport:
    """Assert that the plugin called *name* is installed, loaded and reachable.

    Args:
        name: Entry-point name under the ``drevalpy.plugins`` group.

    Returns:
        A report naming every component the plugin contributed.

    Raises:
        PluginCheckError: If the entry point is undeclared, failed to import, or
            registered no component at all.
    """
    entry_point = _declared_entry_point(name)
    module = _load(entry_point)
    root_package = module.split(".")[0]
    components = _contributed(root_package)
    if not components:
        msg = (
            f"Plugin {name!r} imported cleanly but registered nothing under {root_package!r}. "
            f"Registries checked: {sorted(_REGISTRIES)}. "
            "Check that the entry-point module imports the modules holding the @register decorators."
        )
        raise PluginCheckError(msg)
    return PluginReport(name=name, value=entry_point.value, module=module, components=components)

"""Plugin discovery via importlib.metadata entry points.

Load failures are recorded rather than swallowed: a plugin that fails to import
silently removes every component it would have registered, which surfaces much
later as an "unknown predictor" error far from the cause. Failures are therefore
kept in :func:`get_failed_plugins` and reported at WARNING level, mirroring
:func:`drevalpy.registry._builtins.get_skipped_builtin_modules`.

Set ``DREVALPY_STRICT_PLUGINS=1`` to re-raise instead of warn. Plugin CI wants
this; the default stays non-fatal so one broken third-party package cannot brick
the CLI for everyone else.
"""

from __future__ import annotations

import importlib.metadata
import os
import traceback

from drevalpy.log import get_logger

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "drevalpy.plugins"
STRICT_ENV_VAR = "DREVALPY_STRICT_PLUGINS"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

_discovered = False
_LOADED_PLUGINS: dict[str, str] = {}
_FAILED_PLUGINS: dict[str, str] = {}


def strict_plugins_enabled() -> bool:
    """Return whether plugin load failures should propagate.

    :returns: ``True`` when ``DREVALPY_STRICT_PLUGINS`` is set to a truthy value
        (``1``, ``true``, ``yes`` or ``on``, case-insensitive).
    """
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def get_loaded_plugins() -> dict[str, str]:
    """Return plugins that were imported successfully.

    :returns: Mapping of entry-point name to the entry-point value (the dotted
        object reference the plugin declared). Empty before discovery runs.
    """
    return dict(_LOADED_PLUGINS)


def get_failed_plugins() -> dict[str, str]:
    """Return plugins that raised while being imported.

    :returns: Mapping of entry-point name to the formatted traceback of the
        failure. Empty when every declared plugin loaded cleanly.
    """
    return dict(_FAILED_PLUGINS)


def _load_entry_point(ep: importlib.metadata.EntryPoint) -> None:
    """Import one entry point, recording success or failure.

    :param ep: Entry point declared under the ``drevalpy.plugins`` group.
    :raises Exception: Any error raised by the plugin, when strict mode is on.
    """
    logger.debug("Loading plugin %r", ep.name)
    try:
        ep.load()
    except Exception:
        _LOADED_PLUGINS.pop(ep.name, None)
        _FAILED_PLUGINS[ep.name] = traceback.format_exc()
        if strict_plugins_enabled():
            logger.error("Failed to load drevalpy plugin %r (%s is set)", ep.name, STRICT_ENV_VAR)
            raise
        logger.warning(
            "Failed to load drevalpy plugin %r: its components will be unavailable. "
            "See drevalpy.registry.get_failed_plugins() for the traceback, "
            "or set %s=1 to make this fatal.",
            ep.name,
            STRICT_ENV_VAR,
            exc_info=True,
        )
    else:
        _FAILED_PLUGINS.pop(ep.name, None)
        _LOADED_PLUGINS[ep.name] = getattr(ep, "value", "")


def discover_plugins() -> None:
    """Import installed packages declaring the ``drevalpy.plugins`` entry point.

    Uses the ``importlib.metadata.entry_points`` API to find third-party packages
    that register themselves under the ``drevalpy.plugins`` group.  Each entry point
    is loaded (imported), which triggers any registration decorators the plugin
    defines.

    An idempotency guard ensures this function only performs discovery once per
    process, regardless of how many times it is called. The guard is set before
    loading anything, so a plugin that imports drevalpy does not recurse -- and so
    a strict-mode failure does not re-run discovery on the next call.

    :raises Exception: The first plugin failure, when ``DREVALPY_STRICT_PLUGINS``
        is set. Otherwise failures are recorded in :func:`get_failed_plugins`.
    """
    global _discovered  # noqa: PLW0603
    if _discovered:
        return
    _discovered = True
    logger.debug("Discovering plugins")
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        _load_entry_point(ep)

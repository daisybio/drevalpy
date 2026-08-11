"""Plugin discovery via importlib.metadata entry points."""

from __future__ import annotations

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

_discovered = False


def discover_plugins() -> None:
    """Import installed packages declaring the ``drevalpy.plugins`` entry point.

    Uses the ``importlib.metadata.entry_points`` API to find third-party packages
    that register themselves under the ``drevalpy.plugins`` group.  Each entry point
    is loaded (imported), which triggers any registration decorators the plugin
    defines.

    An idempotency guard ensures this function only performs discovery once per
    process, regardless of how many times it is called.
    """
    global _discovered  # noqa: PLW0603
    if _discovered:
        return
    _discovered = True
    logger.debug("Discovering plugins")
    for ep in importlib.metadata.entry_points(group="drevalpy.plugins"):
        logger.debug("Loading plugin %r", ep.name)
        try:
            ep.load()
        except Exception:
            logger.warning("Failed to load drevalpy plugin %r", ep.name, exc_info=True)

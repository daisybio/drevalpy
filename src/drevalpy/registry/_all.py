"""Every in-memory registry, by kind, for snapshots that must cover all of them.

Extension loading rolls a failed load back through these snapshots, and the test suite
restores them after every test. The dataset registry is not included: it persists its
entries in the user config file rather than in memory.
"""

from __future__ import annotations

from typing import Any

from drevalpy.registry._registry import Registry, Snapshot


def all_registries() -> dict[str, Registry[Any]]:
    """Return every in-memory registry.

    The zoo registry is returned as stored, without parsing the built-in presets: a
    snapshot of the still-empty zoo restores to empty, and the next access parses them.

    :returns: Registries keyed by kind: ``predictor``, ``featurizer``, ``splitter``,
        ``visualization`` and ``zoo``.
    """
    from drevalpy.models import zoo
    from drevalpy.registry.featurizer import featurizers
    from drevalpy.registry.predictor import predictor_registry
    from drevalpy.registry.splitter import splitter_registry
    from drevalpy.registry.visualization import visualization_registry

    return {
        "predictor": predictor_registry,
        "featurizer": featurizers,
        "splitter": splitter_registry,
        "visualization": visualization_registry,
        "zoo": zoo._ZOO,
    }


def snapshot_all() -> dict[str, Snapshot[Any]]:
    """Capture every in-memory registry.

    :returns: One snapshot per registry kind, for :func:`restore_all`.
    """
    return {kind: registry.snapshot() for kind, registry in all_registries().items()}


def restore_all(snapshots: dict[str, Snapshot[Any]]) -> None:
    """Put every in-memory registry back to the state :func:`snapshot_all` captured.

    :param snapshots: Snapshots returned by :func:`snapshot_all`.
    """
    for kind, registry in all_registries().items():
        registry.restore(snapshots[kind])

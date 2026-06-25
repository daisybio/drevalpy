"""Feature kind and contract objects for component compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureKind(StrEnum):
    """Format of featurizer outputs / predictor inputs."""

    DENSE = "dense"
    GRAPH = "graph"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class FeatureContract:
    """Structured description of a feature representation."""

    kind: FeatureKind
    view: str | None = None
    backend: str | None = None
    scope: str | None = None
    has_node_features: bool | None = None
    has_edge_features: bool | None = None


_GRAPH_FIELDS = ("view", "backend", "scope", "has_node_features", "has_edge_features")


def contracts_compatible(produced: FeatureContract, required: FeatureContract) -> bool:
    """Return whether *produced* satisfies *required*.

    For ``DENSE`` and ``SEQUENCE``, matching ``kind`` is sufficient.
    When *required* sets ``scope``, *produced* must match it when present.
    For ``GRAPH``, any non-``None`` field on *required* must match *produced*.
    """
    if produced.kind != required.kind:
        return False
    if produced.kind in {FeatureKind.DENSE, FeatureKind.SEQUENCE}:
        if required.scope is not None and produced.scope is not None and produced.scope != required.scope:
            return False
        return True
    if produced.kind != FeatureKind.GRAPH:
        return True
    for field_name in _GRAPH_FIELDS:
        required_value = getattr(required, field_name)
        if required_value is not None and getattr(produced, field_name) != required_value:
            return False
    return True

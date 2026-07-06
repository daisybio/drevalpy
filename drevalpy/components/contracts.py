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
    """Structured description of a feature representation.

    Compatibility intentionally checks only the broad feature kind for now.
    Additional fields may be added later when real compatibility requirements
    appear; graph compatibility is therefore currently just ``graph`` expected
    and ``graph`` provided.
    """

    kind: FeatureKind


def contracts_compatible(produced: FeatureContract, required: FeatureContract) -> bool:
    """Return whether *produced* satisfies *required*."""
    return produced.kind == required.kind

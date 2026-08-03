"""Typed featurizer block payloads and configuration-time block specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from drevalpy.components.contracts import FeatureFormat


@dataclass(frozen=True)
class BlockSpec:
    """Declare a named block emitted by a featurizer for compatibility checks."""

    name: str
    format: FeatureFormat
    metadata: bool = False


@dataclass(frozen=True)
class FeatureBlock:
    """Named featurizer output with format and optional feature metadata.

    ``values`` is always an ``np.ndarray``. Dense numeric blocks use float arrays;
    graph and ragged blocks use object-dtype arrays whose elements are arbitrary payloads.
    """

    values: np.ndarray
    format: FeatureFormat
    feature_names: tuple[str, ...] | None = None
    metadata: Mapping[str, object] | None = None
    entity_aligned: bool = True

    def __post_init__(self) -> None:
        """Freeze metadata into an immutable mapping when provided."""
        if self.metadata is not None and not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def numeric_feature_block(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FeatureBlock:
    """Build a dense numeric matrix block."""
    return FeatureBlock(
        values=values,
        format=FeatureFormat.NUMERIC_MATRIX,
        feature_names=feature_names,
        metadata=metadata,
    )


def graph_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a graph payload block without dtype coercion."""
    return FeatureBlock(values=values, format=FeatureFormat.GRAPH)


def ragged_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a ragged sequence payload block without dtype coercion."""
    return FeatureBlock(values=values, format=FeatureFormat.RAGGED_SEQUENCE)


def metadata_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a global metadata block that is not indexed per entity."""
    return FeatureBlock(
        values=values,
        format=FeatureFormat.NUMERIC_MATRIX,
        entity_aligned=False,
    )


def merge_feature_blocks(
    *block_maps: Mapping[str, FeatureBlock],
) -> dict[str, FeatureBlock]:
    """Merge child block mappings, rejecting duplicate emitted names."""
    merged: dict[str, FeatureBlock] = {}
    for block_map in block_maps:
        for name, block in block_map.items():
            if name in merged:
                msg = f"Duplicate featurizer block name {name!r}"
                raise ValueError(msg)
            merged[name] = block
    return merged

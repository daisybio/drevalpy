"""Typed featurizer block payloads and configuration-time block specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat


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


def _validate_projection(blocks: Mapping[str, FeatureBlock], projection: tuple[str, ...]) -> None:
    """Require an ordered, unambiguous selection of compatible entity blocks."""
    for name in projection:
        if name not in blocks:
            raise ValueError(f"Matrix projection selects missing block {name!r}")
        block = blocks[name]
        if not block.entity_aligned:
            raise ValueError(f"Matrix projection cannot select global metadata block {name!r}")
        if len(projection) > 1 and (
            block.format != FeatureFormat.NUMERIC_MATRIX
            or block.values.ndim != 2
            or block.values.dtype.kind not in "biufc"
        ):
            raise ValueError("Multiple matrix blocks must be two-dimensional numeric arrays")
    if len(set(projection)) != len(projection):
        raise ValueError("Matrix projection cannot repeat a block")


@dataclass(frozen=True)
class FeaturizerResult:
    """Named feature blocks in entity order, with an explicit matrix selection.

    The projection names blocks in order. One block is passed through unchanged;
    several numeric blocks are concatenated. Global metadata is never projected.
    """

    entity_ids: np.ndarray
    blocks: Mapping[str, FeatureBlock]
    projection: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate row identities, block alignment, and projection compatibility."""
        ids = np.asarray(self.entity_ids)
        if ids.ndim != 1:
            raise ValueError("FeaturizerResult.entity_ids must be one-dimensional")
        object.__setattr__(self, "entity_ids", ids)
        object.__setattr__(self, "blocks", MappingProxyType(dict(self.blocks)))
        object.__setattr__(self, "projection", tuple(self.projection))
        for name, block in self.blocks.items():
            if block.entity_aligned and (block.values.ndim == 0 or len(block.values) != len(ids)):
                raise ValueError(f"Feature block {name!r} rows must match entity_ids")
        _validate_projection(self.blocks, self.projection)

    @cached_property
    def matrix(self) -> np.ndarray:
        """Materialize the declared matrix once, only when a consumer needs it."""
        if not self.projection:
            return np.empty((len(self.entity_ids), 0), dtype=np.float32)
        arrays = [self.blocks[name].values for name in self.projection]
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)

    @classmethod
    def from_blocks(cls, blocks: Mapping[str, FeatureBlock], entity_ids: np.ndarray) -> FeaturizerResult:
        """Explicitly select all ordered numeric entity blocks for concatenation.

        :param blocks: Canonical named payloads, including optional global metadata.
        :param entity_ids: Identifiers in row order.
        :returns: Result used by numeric concatenating featurizers.
        """
        projection = tuple(
            name
            for name, block in blocks.items()
            if block.entity_aligned and block.format == FeatureFormat.NUMERIC_MATRIX
        )
        return cls(entity_ids, blocks, projection)


def numeric_feature_block(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FeatureBlock:
    """Build a dense numeric matrix block.

    :param values: Entity-aligned 2D float array.
    :param feature_names: Optional column names for the matrix.
    :param metadata: Optional per-block metadata mapping.

    :returns: ``FeatureBlock`` with ``FeatureFormat.NUMERIC_MATRIX``.
    """
    return FeatureBlock(
        values=values,
        format=FeatureFormat.NUMERIC_MATRIX,
        feature_names=feature_names,
        metadata=metadata,
    )


def graph_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a graph payload block without dtype coercion.

    :param values: Object-dtype array of graph payloads, one row per entity.

    :returns: ``FeatureBlock`` with ``FeatureFormat.GRAPH``.
    """
    return FeatureBlock(values=values, format=FeatureFormat.GRAPH)


def ragged_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a ragged sequence payload block without dtype coercion.

    :param values: Object-dtype array of variable-length sequence payloads.

    :returns: ``FeatureBlock`` with ``FeatureFormat.RAGGED_SEQUENCE``.
    """
    return FeatureBlock(values=values, format=FeatureFormat.RAGGED_SEQUENCE)


def metadata_feature_block(values: np.ndarray) -> FeatureBlock:
    """Build a global metadata block that is not indexed per entity.

    :param values: Payload array stored once for the whole batch.

    :returns: ``FeatureBlock`` marked with ``entity_aligned=False``.
    """
    return FeatureBlock(
        values=values,
        format=FeatureFormat.NUMERIC_MATRIX,
        entity_aligned=False,
    )


def merge_feature_blocks(
    *block_maps: Mapping[str, FeatureBlock],
) -> dict[str, FeatureBlock]:
    """Merge child block mappings, rejecting duplicate emitted names.

    :param *block_maps: One or more block mappings to combine in order.

    :returns: Single mapping containing every block from ``block_maps``.

    :raises ValueError: If the same block name appears in more than one mapping.
    """
    merged: dict[str, FeatureBlock] = {}
    for block_map in block_maps:
        for name, block in block_map.items():
            if name in merged:
                msg = f"Duplicate featurizer block name {name!r}"
                raise ValueError(msg)
            merged[name] = block
    return merged

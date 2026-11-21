"""Typed featurizer block payloads and entity-aligned featurizer outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class FeatureBlock:
    """Named featurizer output with format and optional annotations.

    ``values`` is an ``np.ndarray``: dense numeric blocks use float arrays, graph and ragged
    blocks use object-dtype arrays of per-entity payloads, and global blocks set
    ``entity_aligned=False``.
    """

    values: np.ndarray
    format: FeatureFormat = FeatureFormat.NUMERIC_MATRIX
    feature_names: tuple[str, ...] | None = None
    annotations: Mapping[str, object] | None = None
    entity_aligned: bool = True


@dataclass(frozen=True)
class FeaturizerOutput:
    """Named feature blocks aligned to unique entities.

    Block predictors read ``blocks[name]`` directly. Matrix predictors read :attr:`matrix`,
    which horizontally concatenates all entity-aligned numeric blocks in insertion order
    (skipping non-numeric and ``entity_aligned=False`` blocks).
    """

    entity_ids: np.ndarray
    blocks: dict[str, FeatureBlock]

    def __post_init__(self) -> None:
        """Validate row identities and entity alignment of every block."""
        ids = np.asarray(self.entity_ids)
        if ids.ndim != 1:
            raise ValueError("FeaturizerOutput.entity_ids must be one-dimensional")
        object.__setattr__(self, "entity_ids", ids)
        object.__setattr__(self, "blocks", dict(self.blocks))
        for name, block in self.blocks.items():
            if block.entity_aligned and (block.values.ndim == 0 or len(block.values) != len(ids)):
                raise ValueError(f"Feature block {name!r} rows must match entity_ids")

    @cached_property
    def matrix(self) -> np.ndarray:
        """Materialize all entity-aligned numeric blocks as one 2D matrix."""
        arrays = [
            block.values
            for block in self.blocks.values()
            if block.entity_aligned and block.format == FeatureFormat.NUMERIC_MATRIX
        ]
        if not arrays:
            return np.empty((len(self.entity_ids), 0), dtype=np.float32)
        for array in arrays:
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
                raise ValueError("Numeric feature blocks must be two-dimensional numeric arrays")
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)

    @cached_property
    def _entity_index(self) -> pd.Index:
        import pandas as pd

        return pd.Index(self.entity_ids)

    def rows_for(self, ids: np.ndarray) -> np.ndarray:
        """Locate the row of each identifier in ``entity_ids``.

        :param ids: Entity identifiers, in the order the rows are wanted.
        :returns: Integer row index per identifier.
        :raises ValueError: If ``entity_ids`` has duplicates or an identifier is missing from it.
        """
        index = self._entity_index
        if not index.is_unique:
            raise ValueError("entity_ids must be unique to look up rows")
        rows = index.get_indexer(ids)
        missing = np.flatnonzero(rows < 0)
        if missing.size:
            preview = ", ".join(repr(str(ids[i])) for i in missing[:5])
            suffix = f" (+{missing.size - 5} more)" if missing.size > 5 else ""
            raise ValueError(f"Missing identifiers in featurizer rows: {preview}{suffix}")
        return rows.astype(np.int64, copy=False)

"""Shared stubs for featurizer tests.

Plain module (no ``__init__.py``) imported by dotted path, per the test layout
rules in ``AGENTS.md``.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource


class StubSource(FeatureSource):
    """Minimal feature source serving one view matrix, NaN rows included."""

    def __init__(self, view_matrix: np.ndarray, identifiers: np.ndarray) -> None:
        """Store the backing matrix and its row identifiers.

        :param view_matrix: Rows aligned with *identifiers*.
        :param identifiers: Entity IDs addressing the matrix rows.
        """
        self._view_matrix = view_matrix
        self._identifiers = identifiers

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return self._identifiers

    @property
    def mdata(self) -> None:
        """No MuData backing for stubs."""
        return None

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return the rows of the backing matrix for *entity_ids*."""
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        indices = [idx_map[eid] for eid in entity_ids]
        return self._view_matrix[indices]

    def get_entity_view(self, entity_id: str, view: str) -> np.ndarray | None:
        """Return one row, or ``None`` for an unknown entity."""
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        if entity_id not in idx_map:
            return None
        return self._view_matrix[idx_map[entity_id]]

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """No feature names for stubs."""
        return None


class DoublingFeaturizer(Featurizer):
    """Featurizer that doubles the values of ``test_view``."""

    input_views = ("test_view",)

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        """Fitting is a no-op; the transform is stateless."""
        return self

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return one numeric block holding the doubled view."""
        return {"test_view": numeric_feature_block(self._transform(source, entity_ids))}

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return the doubled view matrix."""
        matrix = source.get_view_matrix("test_view", entity_ids)
        return (matrix * 2).astype(np.float32)

    @property
    def output_dim(self) -> int:
        """Fixed width of the stub view."""
        return 3


# Registration normally injects the contract; these stubs are never registered.
DoublingFeaturizer.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

"""Tests for NaN tolerance in the Featurizer base class."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from drevalpy.components.core.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.base import Featurizer


class _StubSource(FeatureSource):
    """Minimal feature source that serves a view matrix with NaN rows."""

    def __init__(self, view_matrix: np.ndarray, identifiers: np.ndarray) -> None:
        self._view_matrix = view_matrix
        self._identifiers = identifiers

    @property
    def identifiers(self) -> np.ndarray:
        return self._identifiers

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        indices = [idx_map[eid] for eid in entity_ids]
        return self._view_matrix[indices]

    def get_entity_view(self, entity_id: str, view: str) -> np.ndarray | None:
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        if entity_id not in idx_map:
            return None
        return self._view_matrix[idx_map[entity_id]]

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        return None


class _DoublingFeaturizer(Featurizer):
    """Test featurizer that doubles input values."""

    input_views = ("test_view",)

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        return self

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        matrix = source.get_view_matrix("test_view", entity_ids)
        return {"test_view": numeric_feature_block((matrix * 2).astype(np.float32))}

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        matrix = source.get_view_matrix("test_view", entity_ids)
        return (matrix * 2).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return 3


# Inject a contract (normally done by the registry decorator)
_DoublingFeaturizer.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


@pytest.fixture
def mixed_source() -> tuple[_StubSource, np.ndarray]:
    """Source with 5 entities: first and last are all-NaN."""
    ids = np.array(["A", "B", "C", "D", "E"])
    matrix = np.array(
        [
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    return _StubSource(matrix, ids), ids


@pytest.fixture
def all_nan_source() -> tuple[_StubSource, np.ndarray]:
    """Source where all entities are all-NaN."""
    ids = np.array(["X", "Y", "Z"])
    matrix = np.full((3, 3), np.nan, dtype=np.float32)
    return _StubSource(matrix, ids), ids


@pytest.fixture
def all_valid_source() -> tuple[_StubSource, np.ndarray]:
    """Source where all entities are valid."""
    ids = np.array(["A", "B", "C"])
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    return _StubSource(matrix, ids), ids


class TestTransformNaNTolerance:
    """Tests for the transform() NaN tolerance wrapper."""

    def test_all_valid_passes_through(self, all_valid_source):
        source, ids = all_valid_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_valid_invalid(self, mixed_source):
        source, ids = mixed_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        assert result.shape == (5, 3)
        assert np.all(np.isnan(result[0]))
        assert np.all(np.isnan(result[4]))
        np.testing.assert_array_almost_equal(result[1], [2, 4, 6])
        np.testing.assert_array_almost_equal(result[2], [8, 10, 12])
        np.testing.assert_array_almost_equal(result[3], [14, 16, 18])

    def test_all_nan_produces_nan_output(self, all_nan_source):
        source, ids = all_nan_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        result = feat.transform(source, ids)
        assert result.shape[0] == 3
        assert np.all(np.isnan(result))


class TestTransformBlocksNaNTolerance:
    """Tests for the transform_blocks() NaN tolerance wrapper."""

    def test_all_valid_passes_through(self, all_valid_source):
        source, ids = all_valid_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        assert "test_view" in blocks
        expected = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=np.float32)
        np.testing.assert_array_almost_equal(blocks["test_view"].values, expected)

    def test_mixed_valid_invalid(self, mixed_source):
        source, ids = mixed_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        values = blocks["test_view"].values
        assert values.shape == (5, 3)
        assert np.all(np.isnan(values[0]))
        assert np.all(np.isnan(values[4]))
        np.testing.assert_array_almost_equal(values[1], [2, 4, 6])
        np.testing.assert_array_almost_equal(values[2], [8, 10, 12])
        np.testing.assert_array_almost_equal(values[3], [14, 16, 18])

    def test_all_nan_produces_nan_blocks(self, all_nan_source):
        source, ids = all_nan_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        blocks = feat.transform_blocks(source, ids)
        values = blocks["test_view"].values
        assert values.shape[0] == 3
        assert np.all(np.isnan(values))


class TestNaNWarning:
    """Tests for the warning threshold logic."""

    def test_warning_above_threshold(self, mixed_source, caplog):
        source, ids = mixed_source
        feat = _DoublingFeaturizer()
        feat.nan_threshold = 0.2
        feat.fit(source, entity_ids=ids)
        with caplog.at_level(logging.WARNING, logger="drevalpy.components.featurizers.base"):
            feat.transform(source, ids)
        assert any("invalid" in record.message.lower() for record in caplog.records)

    def test_no_warning_below_threshold(self, mixed_source, caplog):
        source, ids = mixed_source
        feat = _DoublingFeaturizer()
        feat.nan_threshold = 0.5
        feat.fit(source, entity_ids=ids)
        with caplog.at_level(logging.WARNING, logger="drevalpy.components.featurizers.base"):
            feat.transform(source, ids)
        nan_warnings = [r for r in caplog.records if "invalid" in r.message.lower()]
        assert not nan_warnings


class TestConsistency:
    """Verify transform and transform_blocks produce consistent NaN handling."""

    def test_transform_and_blocks_agree(self, mixed_source):
        source, ids = mixed_source
        feat = _DoublingFeaturizer()
        feat.fit(source, entity_ids=ids)
        matrix = feat.transform(source, ids)
        blocks = feat.transform_blocks(source, ids)
        block_values = blocks["test_view"].values
        np.testing.assert_array_equal(
            np.isnan(matrix),
            np.isnan(block_values),
        )
        valid_mask = ~np.isnan(matrix).all(axis=1)
        np.testing.assert_array_almost_equal(matrix[valid_mask], block_values[valid_mask])

"""Tests for SplitMasks, SplitMask, and the splitter system."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from drevalpy.data.splitters import (
    SplitValidationError,
    get_splitter,
    splitter_registry,
)
from drevalpy.data.splitters.validation import validate_folds
from drevalpy.types import SplitMask, SplitMasks

# ------------------------------------------------------------------
# Mock MuDataLike for testing
# ------------------------------------------------------------------


class _MockMuDataset:
    """Minimal MuDataLike for testing splitters."""

    def __init__(self, n_cl: int = 10, n_dr: int = 8, density: float = 0.7, n_tissues: int = 3):
        rng = np.random.default_rng(42)
        self._response = rng.standard_normal((n_cl, n_dr)).astype(np.float32)
        mask = rng.random((n_cl, n_dr)) > density
        self._response[mask] = np.nan
        self._cl_ids = np.array([f"CL_{i}" for i in range(n_cl)])
        self._dr_ids = np.array([f"DR_{i}" for i in range(n_dr)])
        self._tissues = np.array([f"Tissue_{i % n_tissues}" for i in range(n_cl)])

    @property
    def cell_line_ids(self) -> np.ndarray:
        return self._cl_ids

    @property
    def drug_ids(self) -> np.ndarray:
        return self._dr_ids

    @property
    def response_matrix(self) -> np.ndarray:
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        idx_map = {name: i for i, name in enumerate(self._cl_ids)}
        indices = [idx_map[str(x)] for x in ids]
        return self._tissues[indices]


@pytest.fixture
def mock_dataset() -> _MockMuDataset:
    return _MockMuDataset()


# ------------------------------------------------------------------
# SplitMasks tests
# ------------------------------------------------------------------


def _mask(shape: tuple[int, int], *positions: tuple[int, int]) -> SplitMask:
    """Helper to build a SplitMask with True at given positions."""
    m = np.zeros(shape, dtype=bool)
    for r, c in positions:
        m[r, c] = True
    return SplitMask(m)


class TestSplitMasks:
    def test_creation(self):
        shape = (4, 3)
        masks = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1)),
            test=_mask(shape, (2, 0)),
            val=_mask(shape, (3, 1)),
        )
        assert masks.train.shape == shape
        assert masks.test.shape == shape
        assert masks.val.shape == shape
        assert len(masks.train) == 2
        assert len(masks.test) == 1
        assert len(masks.val) == 1

    def test_metadata_default_empty(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        assert masks.metadata == {}

    def test_metadata_mutable(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        masks.metadata["key"] = "value"
        assert masks.metadata["key"] == "value"

    def test_save_load_roundtrip(self):
        shape = (6, 3)
        masks = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1), (2, 2)),
            test=_mask(shape, (3, 0), (4, 1)),
            val=_mask(shape, (5, 2)),
            metadata={"mode": "LCO", "fold_index": 0, "custom": 42},
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        np.testing.assert_array_equal(loaded.train.mask, masks.train.mask)
        np.testing.assert_array_equal(loaded.test.mask, masks.test.mask)
        np.testing.assert_array_equal(loaded.val.mask, masks.val.mask)
        assert loaded.metadata == masks.metadata

    def test_save_load_empty_val(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert not loaded.val.any()
        assert loaded.val.shape == shape

    def test_save_load_no_metadata(self):
        shape = (2, 2)
        masks = SplitMasks(
            train=_mask(shape, (0, 0)),
            test=_mask(shape, (1, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert loaded.metadata == {}


# ------------------------------------------------------------------
# SplitMask tests
# ------------------------------------------------------------------


class TestSplitMask:
    def test_creation_from_mask(self):
        mask = np.array([[True, False], [False, True]])
        scope = SplitMask(mask=mask)
        assert scope.pairs.shape == (2, 2)
        assert len(scope) == 2

    def test_from_pairs(self):
        pairs = np.array([[0, 0], [1, 1]])
        scope = SplitMask.from_pairs(pairs, shape=(2, 2))
        assert scope.mask[0, 0]
        assert scope.mask[1, 1]
        assert not scope.mask[0, 1]
        assert len(scope) == 2

    def test_pairs_property_matches_mask(self):
        mask = np.array([[True, False, True], [False, True, False]])
        scope = SplitMask(mask=mask)
        expected = np.argwhere(mask)
        np.testing.assert_array_equal(scope.pairs, expected)


# ------------------------------------------------------------------
# Splitter registry tests
# ------------------------------------------------------------------


class TestSplitterRegistry:
    def test_builtin_modes_registered(self):
        assert "LPO" in splitter_registry.modes
        assert "LCO" in splitter_registry.modes
        assert "LDO" in splitter_registry.modes
        assert "LTO" in splitter_registry.modes

    def test_get_returns_callable(self):
        splitter = splitter_registry.get("LPO")
        assert callable(splitter)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            splitter_registry.get("NONEXISTENT")

    def test_resolve_string(self):
        splitter = splitter_registry.resolve("LCO")
        assert callable(splitter)

    def test_resolve_callable_passthrough(self):
        def my_fn(mudataset, n_splits=5, validation_ratio=0.1, random_state=42):
            return []

        assert splitter_registry.resolve(my_fn) is my_fn

    def test_get_splitter_alias(self):
        assert get_splitter("LPO") is splitter_registry.get("LPO")

    def test_repr_shows_modes(self):
        output = repr(splitter_registry)
        assert "LPO" in output
        assert "LCO" in output


# ------------------------------------------------------------------
# Splitter function tests
# ------------------------------------------------------------------


class TestLeavePairOut:
    def test_produces_correct_number_of_folds(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_all_folds_are_2d_bool(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        shape = mock_dataset.response_matrix.shape
        for fold in folds:
            assert fold.train.shape == shape
            assert fold.test.shape == shape
            assert fold.val.shape == shape
            assert fold.train.mask.dtype == bool
            assert fold.test.mask.dtype == bool
            assert fold.val.mask.dtype == bool

    def test_no_pair_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            assert not (fold.train & fold.test).any()

    def test_metadata_injected(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for i, fold in enumerate(folds):
            assert fold.metadata["mode"] == "LPO"
            assert fold.metadata["fold_index"] == i
            assert fold.metadata["n_splits"] == 3


class TestLeaveCellLineOut:
    def test_produces_folds(self, mock_dataset):
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_cell_line_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_rows = set(np.where(fold.train.mask.any(axis=1))[0].tolist())
            test_rows = set(np.where(fold.test.mask.any(axis=1))[0].tolist())
            assert train_rows & test_rows == set()

    def test_all_indices_within_bounds(self, mock_dataset):
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        shape = mock_dataset.response_matrix.shape
        for fold in folds:
            assert fold.train.shape == shape
            assert fold.test.shape == shape
            assert fold.val.shape == shape


class TestLeaveDrugOut:
    def test_produces_folds(self, mock_dataset):
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_drug_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_cols = set(np.where(fold.train.mask.any(axis=0))[0].tolist())
            test_cols = set(np.where(fold.test.mask.any(axis=0))[0].tolist())
            assert train_cols & test_cols == set()


class TestLeaveTissueOut:
    def test_produces_folds(self, mock_dataset):
        splitter = splitter_registry.get("LTO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_tissue_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LTO")
        folds = splitter(mock_dataset, n_splits=3)
        tissues = mock_dataset.get_tissue(mock_dataset.cell_line_ids)
        for fold in folds:
            train_rows = np.where(fold.train.mask.any(axis=1))[0]
            test_rows = np.where(fold.test.mask.any(axis=1))[0]
            train_tissues = set(tissues[train_rows].tolist())
            test_tissues = set(tissues[test_rows].tolist())
            assert train_tissues & test_tissues == set()


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


class TestValidation:
    def test_lco_valid_passes(self, mock_dataset):
        folds = splitter_registry.get("LCO")(mock_dataset, n_splits=3)
        validate_folds(folds, "LCO", mock_dataset)

    def test_lco_invalid_raises(self, mock_dataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1)),
            test=_mask(shape, (0, 2)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LCO"):
            validate_folds([bad_fold], "LCO", mock_dataset)

    def test_ldo_invalid_raises(self, mock_dataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=_mask(shape, (0, 0), (1, 0)),
            test=_mask(shape, (2, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LDO"):
            validate_folds([bad_fold], "LDO", mock_dataset)

    def test_lpo_invalid_raises(self, mock_dataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=_mask(shape, (0, 0), (1, 1)),
            test=_mask(shape, (0, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LPO"):
            validate_folds([bad_fold], "LPO", mock_dataset)

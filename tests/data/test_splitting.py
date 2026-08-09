"""Tests for SplitMasks, EntityScope, and the splitter system."""

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
from drevalpy.data.structures import EntityScope, SplitMasks

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


class TestSplitMasks:
    def test_creation(self):
        masks = SplitMasks(
            train=np.array([[0, 0], [1, 1]]),
            test=np.array([[2, 0]]),
            val=np.array([[3, 1]]),
        )
        assert masks.train.shape == (2, 2)
        assert masks.test.shape == (1, 2)
        assert masks.val.shape == (1, 2)

    def test_metadata_default_empty(self):
        masks = SplitMasks(
            train=np.array([[0, 0]]),
            test=np.array([[1, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        assert masks.metadata == {}

    def test_metadata_mutable(self):
        masks = SplitMasks(
            train=np.array([[0, 0]]),
            test=np.array([[1, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        masks.metadata["key"] = "value"
        assert masks.metadata["key"] == "value"

    def test_save_load_roundtrip(self):
        masks = SplitMasks(
            train=np.array([[0, 0], [1, 1], [2, 2]]),
            test=np.array([[3, 0], [4, 1]]),
            val=np.array([[5, 2]]),
            metadata={"mode": "LCO", "fold_index": 0, "custom": 42},
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        np.testing.assert_array_equal(loaded.train, masks.train)
        np.testing.assert_array_equal(loaded.test, masks.test)
        np.testing.assert_array_equal(loaded.val, masks.val)
        assert loaded.metadata == masks.metadata

    def test_save_load_empty_val(self):
        masks = SplitMasks(
            train=np.array([[0, 0]]),
            test=np.array([[1, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert loaded.val.shape == (0, 2)

    def test_save_load_no_metadata(self):
        masks = SplitMasks(
            train=np.array([[0, 0]]),
            test=np.array([[1, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            masks.save(f.name)
            loaded = SplitMasks.load(f.name)

        assert loaded.metadata == {}


# ------------------------------------------------------------------
# EntityScope tests
# ------------------------------------------------------------------


class TestEntityScope:
    def test_creation(self):
        scope = EntityScope(pairs=np.array([[0, 0], [1, 1]]))
        assert scope.pairs.shape == (2, 2)


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

    def test_all_folds_are_2d(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            assert fold.train.ndim == 2 and fold.train.shape[1] == 2
            assert fold.test.ndim == 2 and fold.test.shape[1] == 2
            assert fold.val.ndim == 2 and fold.val.shape[1] == 2

    def test_no_pair_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LPO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_pairs = set(map(tuple, fold.train.tolist()))
            test_pairs = set(map(tuple, fold.test.tolist()))
            assert train_pairs & test_pairs == set()

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
            train_cls = set(fold.train[:, 0].tolist())
            test_cls = set(fold.test[:, 0].tolist())
            assert train_cls & test_cls == set()

    def test_all_pairs_reference_valid_indices(self, mock_dataset):
        splitter = splitter_registry.get("LCO")
        folds = splitter(mock_dataset, n_splits=3)
        n_cl = len(mock_dataset.cell_line_ids)
        n_dr = len(mock_dataset.drug_ids)
        for fold in folds:
            all_pairs = np.concatenate([fold.train, fold.test, fold.val])
            assert np.all(all_pairs[:, 0] < n_cl)
            assert np.all(all_pairs[:, 1] < n_dr)


class TestLeaveDrugOut:
    def test_produces_folds(self, mock_dataset):
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        assert len(folds) == 3

    def test_no_drug_in_both_train_and_test(self, mock_dataset):
        splitter = splitter_registry.get("LDO")
        folds = splitter(mock_dataset, n_splits=3)
        for fold in folds:
            train_drugs = set(fold.train[:, 1].tolist())
            test_drugs = set(fold.test[:, 1].tolist())
            assert train_drugs & test_drugs == set()


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
            train_tissues = set(tissues[fold.train[:, 0]].tolist())
            test_tissues = set(tissues[fold.test[:, 0]].tolist())
            assert train_tissues & test_tissues == set()


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


class TestValidation:
    def test_lco_valid_passes(self, mock_dataset):
        folds = splitter_registry.get("LCO")(mock_dataset, n_splits=3)
        validate_folds(folds, "LCO", mock_dataset)

    def test_lco_invalid_raises(self, mock_dataset):
        bad_fold = SplitMasks(
            train=np.array([[0, 0], [1, 1]]),
            test=np.array([[0, 2]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        with pytest.raises(SplitValidationError, match="LCO"):
            validate_folds([bad_fold], "LCO", mock_dataset)

    def test_ldo_invalid_raises(self, mock_dataset):
        bad_fold = SplitMasks(
            train=np.array([[0, 0], [1, 0]]),
            test=np.array([[2, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        with pytest.raises(SplitValidationError, match="LDO"):
            validate_folds([bad_fold], "LDO", mock_dataset)

    def test_lpo_invalid_raises(self, mock_dataset):
        bad_fold = SplitMasks(
            train=np.array([[0, 0], [1, 1]]),
            test=np.array([[0, 0]]),
            val=np.empty((0, 2), dtype=np.intp),
        )
        with pytest.raises(SplitValidationError, match="LPO"):
            validate_folds([bad_fold], "LPO", mock_dataset)

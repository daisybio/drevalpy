"""Tests for the splitter registry surface and cross-mode fold validation.

The individual splitters live in ``test_lpo.py`` / ``test_lco.py`` /
``test_ldo.py`` / ``test_lto.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter import (
    SplitValidationError,
    splitter_registry,
)
from drevalpy.registry.splitter import get as get_splitter
from drevalpy.registry.splitter._validation import validate_folds
from drevalpy.types import SplitMask, SplitMasks
from tests.data.splitters._helpers import MockMuDataset, build_mask


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


class TestValidation:
    def test_lco_valid_passes(self, mock_dataset: MockMuDataset):
        folds = splitter_registry.get("LCO")(mock_dataset, n_splits=3)
        validate_folds(folds, "LCO", mock_dataset)

    def test_lco_invalid_raises(self, mock_dataset: MockMuDataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=build_mask(shape, (0, 0), (1, 1)),
            test=build_mask(shape, (0, 2)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LCO"):
            validate_folds([bad_fold], "LCO", mock_dataset)

    def test_ldo_invalid_raises(self, mock_dataset: MockMuDataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=build_mask(shape, (0, 0), (1, 0)),
            test=build_mask(shape, (2, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LDO"):
            validate_folds([bad_fold], "LDO", mock_dataset)

    def test_lpo_invalid_raises(self, mock_dataset: MockMuDataset):
        shape = mock_dataset.response_matrix.shape
        bad_fold = SplitMasks(
            train=build_mask(shape, (0, 0), (1, 1)),
            test=build_mask(shape, (0, 0)),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )
        with pytest.raises(SplitValidationError, match="LPO"):
            validate_folds([bad_fold], "LPO", mock_dataset)

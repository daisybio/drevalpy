"""Tests for split validation: the leakage constraints enforced after every split.

Masks are written out by hand rather than produced by a splitter, so each test
states exactly which leakage it does or does not contain.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter._validation import _VALIDATORS, SplitValidationError, validate_folds
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks

_SHAPE = (4, 3)


class _FakeMuDataset:
    """Minimal ``MuDataLike`` stand-in supplying cell-line ids and tissues."""

    def __init__(self, tissues: tuple[str, ...]) -> None:
        self._tissues = np.array(tissues)

    @property
    def cell_line_ids(self) -> np.ndarray:
        """Row identifiers, one per tissue label."""
        return np.array([f"CL_{i}" for i in range(len(self._tissues))])

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers."""
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        """Fully observed response matrix."""
        return np.ones((len(self._tissues), _SHAPE[1]))

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return the tissue label per requested cell-line id."""
        return self._tissues

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        return ["relevance_score", "fold_change"]

    def get_response_layer(self, name: str) -> np.ndarray:
        """Quality layers on which every curve passes the default thresholds."""
        shape = (len(self._tissues), _SHAPE[1])
        return np.full(shape, 9.0 if name == "relevance_score" else -2.0)


def _mask(rows: tuple[int, ...], cols: tuple[int, ...]) -> SplitMask:
    array = np.zeros(_SHAPE, dtype=bool)
    for row in rows:
        for col in cols:
            array[row, col] = True
    return SplitMask(array)


def _fold(train: SplitMask, test: SplitMask) -> SplitMasks:
    return SplitMasks(train=train, test=test, val=SplitMask(np.zeros(_SHAPE, dtype=bool)))


@pytest.fixture
def mudataset() -> _FakeMuDataset:
    return _FakeMuDataset(("lung", "lung", "skin", "skin"))


def test_split_validation_error_is_a_value_error() -> None:
    assert issubclass(SplitValidationError, ValueError)


def test_every_declared_mode_has_a_validator() -> None:
    assert set(_VALIDATORS) == {"LCO", "LDO", "LPO", "LTO"}


def test_unknown_validation_mode_is_rejected(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0,)), _mask((1,), (0,)))

    with pytest.raises(KeyError):
        validate_folds([fold], "NOPE", mudataset)  # type: ignore[arg-type]


def test_no_folds_is_vacuously_valid(mudataset: _FakeMuDataset) -> None:
    validate_folds([], "LPO", mudataset)


@pytest.mark.parametrize(
    ("mode", "train", "test"),
    [
        pytest.param("LCO", _mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)), id="lco-disjoint-rows"),
        pytest.param("LDO", _mask((0, 1, 2, 3), (0,)), _mask((0, 1, 2, 3), (1, 2)), id="ldo-disjoint-cols"),
        pytest.param("LPO", _mask((0, 1), (0,)), _mask((2, 3), (1,)), id="lpo-disjoint-pairs"),
        pytest.param("LTO", _mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)), id="lto-disjoint-tissues"),
    ],
)
def test_valid_folds_pass(mode: str, train: SplitMask, test: SplitMask, mudataset: _FakeMuDataset) -> None:
    validate_folds([_fold(train, test)], mode, mudataset)  # type: ignore[arg-type]


def test_lco_rejects_a_cell_line_in_both_sides(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0, 1), (0,)), _mask((1, 2), (1,)))

    with pytest.raises(SplitValidationError, match="LCO validation failed"):
        validate_folds([fold], "LCO", mudataset)


def test_ldo_rejects_a_drug_in_both_sides(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0, 1)), _mask((1,), (1, 2)))

    with pytest.raises(SplitValidationError, match="LDO validation failed"):
        validate_folds([fold], "LDO", mudataset)


def test_lto_rejects_a_tissue_in_both_sides(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0, 1, 2)), _mask((1,), (0, 1, 2)))

    with pytest.raises(SplitValidationError, match="LTO validation failed"):
        validate_folds([fold], "LTO", mudataset)


def test_lto_names_the_overlapping_tissue(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0,)), _mask((1,), (1,)))

    with pytest.raises(SplitValidationError, match=r"\['lung'\]"):
        validate_folds([fold], "LTO", mudataset)


def test_lpo_rejects_a_pair_in_both_sides(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0, 1)), _mask((0,), (1, 2)))

    with pytest.raises(SplitValidationError, match="LPO validation failed"):
        validate_folds([fold], "LPO", mudataset)


def test_lpo_tolerates_a_shared_row_and_column(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0,)), _mask((0,), (1,)))

    validate_folds([fold], "LPO", mudataset)


def test_the_failing_fold_index_is_reported(mudataset: _FakeMuDataset) -> None:
    valid = _fold(_mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)))
    leaking = _fold(_mask((0,), (0,)), _mask((0,), (0,)))

    with pytest.raises(SplitValidationError, match=r"fold 1"):
        validate_folds([valid, leaking], "LPO", mudataset)

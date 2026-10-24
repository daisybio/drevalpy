"""Tests for split validation: hand-written masks, so each test states exactly which leakage it contains."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter._validation import _VALIDATORS, SplitValidationError, validate_folds
from drevalpy.types.data.split_mask import SplitMask
from drevalpy.types.data.split_masks import SplitMasks

_SHAPE = (4, 3)


class _FakeMuDataset:
    """Minimal ``MuDataLike`` stand-in supplying sample ids and tissues."""

    def __init__(self, tissues: tuple[str, ...], studies: tuple[str, ...] | None = None) -> None:
        self._tissues = np.array(tissues)
        self._studies = None if studies is None else np.array(studies)

    @property
    def sample_ids(self) -> np.ndarray:
        return np.array([f"CL_{i}" for i in range(len(self._tissues))])

    @property
    def drug_ids(self) -> np.ndarray:
        return np.array([f"D_{i}" for i in range(_SHAPE[1])])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.ones((len(self._tissues), _SHAPE[1]))

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return self._tissues

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        if column == "study" and self._studies is not None:
            return self._studies
        if column != "tissue":
            raise KeyError(f"Sample column '{column}' not found. Available: ['tissue']")
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        return ["relevance_score", "fold_change"]

    def get_response_layer(self, name: str) -> np.ndarray:
        shape = (len(self._tissues), _SHAPE[1])
        return np.full(shape, 9.0 if name == "relevance_score" else -2.0)


def _mask(rows: tuple[int, ...], cols: tuple[int, ...]) -> SplitMask:
    array = np.zeros(_SHAPE, dtype=bool)
    for row in rows:
        for col in cols:
            array[row, col] = True
    return SplitMask(array)


def _fold(train: SplitMask, test: SplitMask, val: SplitMask | None = None) -> SplitMasks:
    return SplitMasks(train=train, test=test, val=val if val is not None else SplitMask(np.zeros(_SHAPE, dtype=bool)))


@pytest.fixture
def mudataset() -> _FakeMuDataset:
    return _FakeMuDataset(("lung", "lung", "skin", "skin"))


@pytest.fixture
def studies() -> _FakeMuDataset:
    return _FakeMuDataset(("lung",) * 4, studies=("GDSC2", "GDSC2", "CTRPv2", "CTRPv2"))


def test_validators_cover_every_declared_mode_and_errors_are_value_errors() -> None:
    assert issubclass(SplitValidationError, ValueError)
    assert set(_VALIDATORS) == {"LSO", "LDO", "LPO", "LTO", "LStO"}


def test_lsto_accepts_a_test_study_absent_from_train_and_validation_even_when_train_and_validation_share_one(
    studies: _FakeMuDataset,
) -> None:
    fold = _fold(_mask((0,), (0, 1)), _mask((2, 3), (0, 1, 2)), _mask((1,), (0, 2)))
    other_way = _fold(_mask((2,), (0, 1)), _mask((0, 1), (0, 1, 2)), _mask((3,), (0,)))

    validate_folds([fold, other_way], "LStO", studies)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("train", "test", "val", "match"),
    [
        (_mask((0, 2), (0,)), _mask((3,), (1,)), None, r"LStO validation failed \(fold 0\).*CTRPv2"),
        (_mask((0,), (0,)), _mask((1, 3), (1,)), _mask((2,), (2,)), "LStO validation failed"),
    ],
    ids=["train-test", "validation-test"],
)
def test_lsto_rejects_a_study_shared_with_test(studies: _FakeMuDataset, train, test, val, match: str) -> None:
    with pytest.raises(SplitValidationError, match=match):
        validate_folds([_fold(train, test, val)], "LStO", studies)  # type: ignore[arg-type]


def test_the_legacy_lco_validation_name_means_lso(mudataset: _FakeMuDataset) -> None:
    leaking = _fold(_mask((0, 1), (0,)), _mask((1, 2), (1,)))

    with pytest.raises(SplitValidationError, match="LSO validation failed"):
        validate_folds([leaking], "LCO", mudataset)  # type: ignore[arg-type]


def test_unknown_validation_mode_is_rejected_and_no_folds_is_vacuously_valid(mudataset: _FakeMuDataset) -> None:
    with pytest.raises(KeyError):
        validate_folds([_fold(_mask((0,), (0,)), _mask((1,), (0,)))], "NOPE", mudataset)  # type: ignore[arg-type]

    validate_folds([], "LPO", mudataset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "train", "test"),
    [
        pytest.param("LSO", _mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)), id="lso-disjoint-rows"),
        pytest.param("LDO", _mask((0, 1, 2, 3), (0,)), _mask((0, 1, 2, 3), (1, 2)), id="ldo-disjoint-cols"),
        pytest.param("LPO", _mask((0, 1), (0,)), _mask((2, 3), (1,)), id="lpo-disjoint-pairs"),
        pytest.param("LPO", _mask((0,), (0,)), _mask((0,), (1,)), id="lpo-shared-row-and-column"),
        pytest.param("LTO", _mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)), id="lto-disjoint-tissues"),
    ],
)
def test_valid_folds_pass(mode: str, train: SplitMask, test: SplitMask, mudataset: _FakeMuDataset) -> None:
    validate_folds([_fold(train, test)], mode, mudataset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "train", "test"),
    [
        pytest.param("LSO", _mask((0, 1), (0,)), _mask((1, 2), (1,)), id="sample"),
        pytest.param("LDO", _mask((0,), (0, 1)), _mask((1,), (1, 2)), id="drug"),
        pytest.param("LTO", _mask((0,), (0, 1, 2)), _mask((1,), (0, 1, 2)), id="tissue"),
        pytest.param("LPO", _mask((0,), (0, 1)), _mask((0,), (1, 2)), id="pair"),
    ],
)
def test_train_test_leakage_is_rejected(mode: str, train: SplitMask, test: SplitMask, mudataset) -> None:
    with pytest.raises(SplitValidationError, match=f"{mode} validation failed"):
        validate_folds([_fold(train, test)], mode, mudataset)  # type: ignore[arg-type]


def test_lto_names_the_overlapping_tissue(mudataset: _FakeMuDataset) -> None:
    fold = _fold(_mask((0,), (0,)), _mask((1,), (1,)))

    with pytest.raises(SplitValidationError, match=r"\['lung'\]"):
        validate_folds([fold], "LTO", mudataset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "train", "test", "val"),
    [
        pytest.param("LSO", _mask((0,), (0,)), _mask((2,), (0,)), _mask((0,), (1,)), id="lso-row"),
        pytest.param("LDO", _mask((0,), (0,)), _mask((1,), (2,)), _mask((2,), (0,)), id="ldo-column"),
        pytest.param("LTO", _mask((0,), (0,)), _mask((2,), (0,)), _mask((1,), (1,)), id="lto-tissue"),
        pytest.param("LPO", _mask((0,), (0,)), _mask((1,), (1,)), _mask((0,), (0,)), id="lpo-pair"),
    ],
)
def test_validation_rejects_validation_leakage(
    mode: str, train: SplitMask, test: SplitMask, val: SplitMask, mudataset: _FakeMuDataset
) -> None:
    with pytest.raises(SplitValidationError, match="validation failed"):
        validate_folds([_fold(train, test, val)], mode, mudataset)  # type: ignore[arg-type]


def test_validation_rejects_a_mask_with_the_wrong_shape(mudataset: _FakeMuDataset) -> None:
    wrong_shape = SplitMask(np.zeros((3, 3), dtype=bool))

    with pytest.raises(SplitValidationError, match="response shape"):
        validate_folds([_fold(_mask((0,), (0,)), _mask((1,), (1,)), wrong_shape)], "LPO", mudataset)  # type: ignore[arg-type]


def test_the_failing_fold_index_is_reported(mudataset: _FakeMuDataset) -> None:
    valid = _fold(_mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)))
    leaking = _fold(_mask((0,), (0,)), _mask((0,), (0,)))

    with pytest.raises(SplitValidationError, match=r"fold 1"):
        validate_folds([valid, leaking], "LPO", mudataset)  # type: ignore[arg-type]

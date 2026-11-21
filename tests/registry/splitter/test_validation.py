"""Tests for fold validation: hand-written folds, so each test states exactly which leakage it contains."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter._validation import SplitValidationError, validate_folds, validate_parts
from drevalpy.types.data.fold import Fold
from drevalpy.types.data.selection_mask import SelectionMask
from tests._fake_dataset import FakeResponseDataset

_SHAPE = (4, 3)


def _dataset(
    tissues: tuple[str, ...],
    studies: tuple[str, ...] | None = None,
    source_samples: tuple[str, ...] | None = None,
) -> FakeResponseDataset:
    """A fake with one row per tissue; *studies* and *source_samples*, when given, are the matching columns."""
    columns = {"study": studies, "source_sample_id": source_samples}
    return FakeResponseDataset(
        shape=(len(tissues), _SHAPE[1]),
        density=1.0,
        tissues=tissues,
        columns={name: values for name, values in columns.items() if values is not None},
    )


def _mask(rows: tuple[int, ...], cols: tuple[int, ...]) -> SelectionMask:
    array = np.zeros(_SHAPE, dtype=bool)
    for row in rows:
        for col in cols:
            array[row, col] = True
    return SelectionMask(array)


def _fold(train: SelectionMask, test: SelectionMask, val: SelectionMask | None = None) -> Fold:
    return Fold(train=train, test=test, val=val if val is not None else SelectionMask(np.zeros(_SHAPE, dtype=bool)))


@pytest.fixture
def dataset() -> FakeResponseDataset:
    return _dataset(("lung", "lung", "skin", "skin"))


@pytest.fixture
def studies() -> FakeResponseDataset:
    return _dataset(("lung",) * 4, studies=("GDSC2", "GDSC2", "CTRPv2", "CTRPv2"))


@pytest.fixture
def concatenated() -> FakeResponseDataset:
    """Rows 0 and 2 are one source sample measured in two studies."""
    return _dataset(
        ("lung", "lung", "skin", "skin"),
        studies=("GDSC2", "GDSC2", "CTRPv2", "CTRPv2"),
        source_samples=("S0", "S1", "S0", "S2"),
    )


def test_errors_are_value_errors() -> None:
    assert issubclass(SplitValidationError, ValueError)


def test_lsto_accepts_a_test_study_absent_from_train_and_validation_even_when_train_and_validation_share_one(
    studies: FakeResponseDataset,
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
def test_lsto_rejects_a_study_shared_with_test(studies: FakeResponseDataset, train, test, val, match: str) -> None:
    with pytest.raises(SplitValidationError, match=match):
        validate_folds([_fold(train, test, val)], "LStO", studies)  # type: ignore[arg-type]


def test_no_folds_is_vacuously_valid(dataset: FakeResponseDataset) -> None:
    validate_folds([], "LPO", dataset)  # type: ignore[arg-type]


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
def test_valid_folds_pass(mode: str, train: SelectionMask, test: SelectionMask, dataset: FakeResponseDataset) -> None:
    validate_folds([_fold(train, test)], mode, dataset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "train", "test"),
    [
        pytest.param("LSO", _mask((0, 1), (0,)), _mask((1, 2), (1,)), id="sample"),
        pytest.param("LDO", _mask((0,), (0, 1)), _mask((1,), (1, 2)), id="drug"),
        pytest.param("LTO", _mask((0,), (0, 1, 2)), _mask((1,), (0, 1, 2)), id="tissue"),
        pytest.param("LPO", _mask((0,), (0, 1)), _mask((0,), (1, 2)), id="pair"),
    ],
)
def test_train_test_leakage_is_rejected(mode: str, train: SelectionMask, test: SelectionMask, dataset) -> None:
    with pytest.raises(SplitValidationError, match=f"{mode} validation failed"):
        validate_folds([_fold(train, test)], mode, dataset)  # type: ignore[arg-type]


def test_lto_names_the_overlapping_tissue(dataset: FakeResponseDataset) -> None:
    fold = _fold(_mask((0,), (0,)), _mask((1,), (1,)))

    with pytest.raises(SplitValidationError, match=r"\['lung'\]"):
        validate_folds([fold], "LTO", dataset)  # type: ignore[arg-type]


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
    mode: str, train: SelectionMask, test: SelectionMask, val: SelectionMask, dataset: FakeResponseDataset
) -> None:
    with pytest.raises(SplitValidationError, match="validation failed"):
        validate_folds([_fold(train, test, val)], mode, dataset)  # type: ignore[arg-type]


def test_validation_rejects_a_mask_with_the_wrong_shape(dataset: FakeResponseDataset) -> None:
    wrong_shape = SelectionMask(np.zeros((3, 3), dtype=bool))

    with pytest.raises(SplitValidationError, match="response shape"):
        validate_folds([_fold(_mask((0,), (0,)), _mask((1,), (1,)), wrong_shape)], "LPO", dataset)  # type: ignore[arg-type]


def test_the_failing_fold_index_is_reported(dataset: FakeResponseDataset) -> None:
    valid = _fold(_mask((0, 1), (0, 1, 2)), _mask((2, 3), (0, 1, 2)))
    leaking = _fold(_mask((0,), (0,)), _mask((0,), (0,)))

    with pytest.raises(SplitValidationError, match=r"fold 1"):
        validate_folds([valid, leaking], "LPO", dataset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "train", "test", "match"),
    [
        pytest.param("LSO", _mask((0,), (0,)), _mask((2,), (1,)), r"1 samples .*\['S0'\]", id="lso-source-sample"),
        pytest.param("LPO", _mask((0,), (1,)), _mask((2,), (1,)), r"\['\(S0, D_1\)'\]", id="lpo-source-pair"),
    ],
)
def test_a_source_sample_measured_in_two_studies_may_not_straddle_two_parts(
    concatenated: FakeResponseDataset, mode: str, train: SelectionMask, test: SelectionMask, match: str
) -> None:
    with pytest.raises(SplitValidationError, match=match):
        validate_folds([_fold(train, test)], mode, concatenated)  # type: ignore[arg-type]


def test_other_source_samples_and_drugs_of_a_concatenated_dataset_pass(concatenated: FakeResponseDataset) -> None:
    validate_folds([_fold(_mask((0, 1), (0,)), _mask((2,), (1,)))], "LPO", concatenated)  # type: ignore[arg-type]
    validate_folds([_fold(_mask((0, 2), (0,)), _mask((1, 3), (1,)))], "LSO", concatenated)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "stopping", "match"),
    [
        pytest.param("LDO", _mask((0,), (1,)), r"drugs appear in both stopping and validation: \['D_1'\]", id="ldo"),
        pytest.param("LTO", _mask((2,), (0,)), r"tissues appear in both stopping and validation: \['skin'\]", id="lto"),
    ],
)
def test_validate_parts_rejects_a_stopping_part_that_shares_a_group_with_validation(
    dataset: FakeResponseDataset, mode: str, stopping: SelectionMask, match: str
) -> None:
    parts = {
        "train": _mask((0,), (0,)) if mode == "LDO" else _mask((0, 1), (0, 1, 2)),
        "stopping": stopping,
        "validation": _mask((1,), (1,)) if mode == "LDO" else _mask((3,), (0,)),
        "test": _mask((0,), (2,)) if mode == "LDO" else SelectionMask(np.zeros(_SHAPE, dtype=bool)),
    }

    with pytest.raises(SplitValidationError, match=r"\(early stopping\).*" + match):
        validate_parts(parts, mode, dataset, context="early stopping")  # type: ignore[arg-type]


def test_validate_parts_accepts_lsto_parts_that_share_samples_outside_the_test_study(
    studies: FakeResponseDataset,
) -> None:
    parts = {
        "train": _mask((0,), (0,)),
        "stopping": _mask((0,), (1,)),
        "validation": _mask((0, 1), (2,)),
        "test": _mask((2, 3), (0, 1, 2)),
    }

    validate_parts(parts, "LStO", studies, context="early stopping")  # type: ignore[arg-type]

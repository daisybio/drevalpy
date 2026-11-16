"""Tests for :func:`drevalpy.registry.splitter._groups.pair_groups`, the leakage unit of each split mode."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.splitter._groups import in_groups, pair_groups


class _Dataset:
    """Three rows by two drugs; rows 0 and 2 are one source sample when *source_samples* is given."""

    def __init__(self, source_samples: tuple[str, ...] | None = None) -> None:
        self._columns = {"tissue": np.array(["skin", "lung", "skin"]), "study": np.array(["B", "A", "A"])}
        if source_samples is not None:
            self._columns["source_sample_id"] = np.array(source_samples)

    @property
    def sample_ids(self) -> np.ndarray:
        return np.array(["X::S0", "X::S1", "Y::S0"])

    @property
    def response_matrix(self) -> np.ndarray:
        return np.ones((3, 2))

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        return self.sample_column("tissue", ids)

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        if column not in self._columns:
            raise KeyError(f"Sample column '{column}' not found.")
        return self._columns[column]


CONCATENATED = _Dataset(source_samples=("S0", "S1", "S0"))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param("LSO", [[0, 0], [1, 1], [0, 0]], id="source-sample"),
        pytest.param("LDO", [[0, 1], [0, 1], [0, 1]], id="drug"),
        pytest.param("LTO", [[1, 1], [0, 0], [1, 1]], id="tissue-in-sorted-order"),
        pytest.param("LStO", [[1, 1], [0, 0], [0, 0]], id="study-in-sorted-order"),
        pytest.param("LPO", [[0, 1], [2, 3], [0, 1]], id="source-sample-and-drug"),
    ],
)
def test_the_unit_of_each_mode(mode: str, expected: list[list[int]]) -> None:
    groups = pair_groups(CONCATENATED, mode)  # type: ignore[arg-type]

    assert groups.dtype == np.int64
    np.testing.assert_array_equal(groups, expected)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("LSO", [[0, 0], [1, 1], [2, 2]]), ("LPO", [[0, 1], [2, 3], [4, 5]])],
)
def test_without_a_source_sample_column_each_row_is_its_own_sample(mode: str, expected: list[list[int]]) -> None:
    np.testing.assert_array_equal(pair_groups(_Dataset(), mode), expected)  # type: ignore[arg-type]


def test_source_samples_are_numbered_by_first_appearance() -> None:
    np.testing.assert_array_equal(pair_groups(_Dataset(("S9", "S1", "S9")), "LSO")[:, 0], [0, 1, 0])  # type: ignore[arg-type]


def test_in_groups_is_isin() -> None:
    groups = pair_groups(CONCATENATED, "LPO")  # type: ignore[arg-type]

    np.testing.assert_array_equal(in_groups(groups, np.array([3, 0])), np.isin(groups, [3, 0]))

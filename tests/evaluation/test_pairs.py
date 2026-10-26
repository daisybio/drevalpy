"""Tests for ``drevalpy.evaluation._pairs``: the constant rule, ``Pairs`` and ``group_pairs``."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drevalpy.evaluation._pairs import CONSTANT_ATOL, Pairs, group_pairs, is_constant


class TestIsConstant:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (np.full(5, 2.5), True),
            (np.array([0.0, 5e-7]), True),
            (np.array([0.0, 2e-6]), False),
            (np.array([1e5, 1e5 + 0.5]), True),
            (np.array([1e5, 1e5 + 5.0]), False),
            (np.array([-1e5, -1e5 + 0.5]), True),
            (np.array([1.0, 2.0, 3.0]), False),
            (np.array([7.0]), True),
        ],
    )
    def test_edge_cases(self, values: np.ndarray, expected: bool) -> None:
        assert is_constant(values) is expected

    def test_the_tolerance_is_absolute_one_micro(self) -> None:
        assert CONSTANT_ATOL == 1e-6
        assert is_constant(np.array([0.0, CONSTANT_ATOL / 10]))

    def test_the_verdict_does_not_depend_on_order(self) -> None:
        """Comparing every value with ``values[0]`` used to make the verdict depend on the first one."""
        rng = np.random.default_rng(0)
        for values in (np.array([0.0, 5e-7, 5e-7, 0.0]), 1e5 + np.array([0.0, 0.9, 0.3]), np.array([-1e5, 1e5, 0.0])):
            assert len({is_constant(rng.permutation(values)) for _ in range(30)}) == 1


class TestPairs:
    def test_finite_drops_pairs_with_a_non_finite_value(self) -> None:
        pairs = Pairs.finite([1.0, np.nan, 3.0, np.inf, 5.0], [1.0, 2.0, -np.inf, 4.0, 5.0])

        assert pairs.y_pred.tolist() == [1.0, 5.0]
        assert pairs.y_true.tolist() == [1.0, 5.0]
        assert len(pairs) == 2

    def test_finite_converts_to_float64(self) -> None:
        pairs = Pairs.finite([1, 2], np.array([3, 4], dtype=np.int8))

        assert pairs.y_pred.dtype == pairs.y_true.dtype == np.float64

    def test_finite_may_leave_no_pairs(self) -> None:
        assert len(Pairs.finite([np.nan], [1.0])) == 0
        assert len(Pairs.finite([], [])) == 0

    def test_finite_rejects_inputs_that_are_not_one_dimensional_and_equally_long(self) -> None:
        with pytest.raises(ValueError, match="equally long"):
            Pairs.finite(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="equally long"):
            Pairs.finite(np.ones((2, 1)), np.ones((2, 1)))

    def test_pairs_are_frozen(self) -> None:
        pairs = Pairs.finite([1.0], [1.0])

        with pytest.raises(dataclasses.FrozenInstanceError):
            pairs.y_pred = np.zeros(1)  # type: ignore[misc]

    def test_constancy_is_decided_per_side_and_computed_once(self) -> None:
        pairs = Pairs.finite([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

        assert "prediction_is_constant" not in vars(pairs)
        assert pairs.prediction_is_constant is True
        assert pairs.response_is_constant is False
        assert {"prediction_is_constant", "response_is_constant"} <= set(vars(pairs))


class TestGroupPairs:
    def test_the_group_axis_is_the_sorted_distinct_labels(self) -> None:
        axis, per_group = group_pairs(np.array(["b", "a", "b", "c"]), np.arange(4.0), np.arange(4.0))

        assert axis.tolist() == ["a", "b", "c"]
        assert [len(pairs) for pairs in per_group] == [1, 2, 1]

    def test_pairs_with_a_non_finite_value_are_dropped(self) -> None:
        labels = np.array(["a", "a", "a", "b", "b", "c"])
        y_pred = np.array([1.0, np.nan, 3.0, np.inf, 5.0, 6.0])
        y_true = np.array([1.0, 2.0, -np.inf, 4.0, 5.0, np.nan])

        axis, per_group = group_pairs(labels, y_pred, y_true)

        assert axis.tolist() == ["a", "b"]
        assert [len(pairs) for pairs in per_group] == [1, 1]
        assert [pairs.y_pred.tolist() for pairs in per_group] == [[1.0], [5.0]]
        assert [pairs.y_true.tolist() for pairs in per_group] == [[1.0], [5.0]]

    def test_a_fixed_axis_drops_unknown_labels_and_keeps_empty_groups(self) -> None:
        labels = np.array(["0", "A", "B", "B", "Z", "A"])
        values = np.array([9.0, 1.0, 2.0, 3.0, 9.0, 4.0])

        axis, per_group = group_pairs(labels, values, values, groups=np.array(["A", "B", "C"]))

        assert axis.tolist() == ["A", "B", "C"]
        assert [len(pairs) for pairs in per_group] == [2, 2, 0]
        assert [pairs.y_pred.tolist() for pairs in per_group] == [[1.0, 4.0], [2.0, 3.0], []]

    def test_a_fixed_axis_without_groups_keeps_no_pairs(self) -> None:
        axis, per_group = group_pairs(np.array(["a"]), np.array([1.0]), np.array([1.0]), groups=np.array([], "U1"))

        assert axis.size == 0
        assert per_group == []

    def test_a_group_holds_its_pairs_in_input_order(self) -> None:
        _, per_group = group_pairs(
            np.array([1, 0, 1, 0]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0])
        )

        assert [(pairs.y_pred.tolist(), pairs.y_true.tolist()) for pairs in per_group] == [
            ([2.0, 4.0], [6.0, 8.0]),
            ([1.0, 3.0], [5.0, 7.0]),
        ]

    def test_inputs_must_be_one_dimensional_and_equally_long(self) -> None:
        with pytest.raises(ValueError, match="equally long"):
            group_pairs(np.array(["a", "b"]), np.array([1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="equally long"):
            group_pairs(np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)))

    def test_each_group_judges_constancy_on_its_own_values(self) -> None:
        _, per_group = group_pairs(
            np.repeat([0, 1], 3), np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0]), np.array([0.0, 5e-7, 0.0, 1.0, 2.0, 3.0])
        )

        assert [pairs.prediction_is_constant for pairs in per_group] == [True, False]
        assert [pairs.response_is_constant for pairs in per_group] == [True, False]

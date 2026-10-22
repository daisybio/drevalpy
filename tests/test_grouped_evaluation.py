"""Tests for :mod:`drevalpy._grouped_evaluation`, the per-group metric rule.

The vectorised paths exist because looping the registered metrics over hundreds of groups
costs seconds per run, so every one of them is pinned against that loop. The rule for
undefined groups is the contract: ``NaN``, never ``0`` or ``1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy._grouped_evaluation import (
    CONSTANT_TOLERANCE,
    GROUPINGS,
    _group_score,
    defined_labels,
    group_labels,
    grouped_scores,
    mean_over_groups,
)
from drevalpy.evaluation import AVAILABLE_METRICS
from tests.synthetic import make_run_result


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels, predictions and response of six groups, with ties in both variables."""
    rng = np.random.default_rng(3)
    labels = np.repeat([f"D{i}" for i in range(6)], 15)
    response = np.round(rng.normal(size=labels.size), 1)
    predictions = np.round(response + rng.normal(size=labels.size), 1)
    return labels, predictions, response


class TestGroupLabels:
    def test_drug_sample_and_study_are_the_supported_groupings(self):
        assert GROUPINGS == ("drug", "sample", "study")

    def test_drug_grouping_reads_drug_ids(self):
        run = make_run_result(n_pairs=6)

        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)

    def test_sample_grouping_reads_sample_ids(self):
        run = make_run_result(n_pairs=6)

        np.testing.assert_array_equal(group_labels(run, "sample"), run.sample_ids)

    def test_study_grouping_reads_the_prefix_before_the_separator(self):
        run = make_run_result(n_pairs=4)
        run.sample_ids = np.array(["GDSC2::CL_1", "CTRPv2::CL_1", "GDSC2::a::b", "CTRPv2::CL_2"])

        assert group_labels(run, "study").tolist() == ["GDSC2", "CTRPv2", "GDSC2", "CTRPv2"]

    @pytest.mark.parametrize("sample_id", ["CL_1", "::CL_1"])
    def test_a_sample_id_without_a_study_prefix_has_no_study(self, sample_id):
        run = make_run_result(n_pairs=2)
        run.sample_ids = np.array(["GDSC2::CL_1", sample_id])

        labels = group_labels(run, "study")

        assert labels[0] == "GDSC2"
        assert labels[1] is np.nan
        assert defined_labels(labels).tolist() == [True, False]

    def test_plain_sample_ids_leave_the_study_grouping_all_undefined(self):
        labels = group_labels(make_run_result(n_pairs=6), "study")

        assert not defined_labels(labels).any()

    def test_drug_and_sample_labels_are_always_defined(self):
        run = make_run_result(n_pairs=6)

        assert defined_labels(group_labels(run, "drug")).all()
        assert defined_labels(group_labels(run, "sample")).all()

    def test_an_unknown_grouping_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            group_labels(make_run_result(), "tissue")


class TestGroupedScores:
    def test_pairs_without_a_group_are_dropped(self):
        labels = np.array(["A", "A", np.nan, "B", "B", np.nan], dtype=object)
        predictions = np.array([1.0, 2.0, 100.0, 1.0, 3.0, -100.0])
        response = np.array([1.0, 2.0, 0.0, 1.0, 2.0, 0.0])

        groups, scores = grouped_scores(labels, predictions, response, ["MSE"])

        assert groups.tolist() == ["A", "B"]
        np.testing.assert_allclose(scores["MSE"], [0.0, 0.5])

    def test_labels_that_are_all_undefined_leave_no_groups(self):
        labels = np.array([np.nan, np.nan], dtype=object)

        groups, scores = grouped_scores(labels, np.array([1.0, 2.0]), np.array([1.0, 3.0]), ["MSE"])

        assert groups.size == 0
        assert scores["MSE"].size == 0

    @pytest.mark.parametrize("metric", list(AVAILABLE_METRICS))
    def test_every_metric_matches_the_registered_function_per_group(self, pairs, metric):
        labels, predictions, response = pairs

        groups, scores = grouped_scores(labels, predictions, response, [metric])

        for index, group in enumerate(groups):
            mask = labels == group
            expected = _group_score(metric, predictions[mask], response[mask])
            assert scores[metric][index] == pytest.approx(expected, abs=1e-9)

    def test_the_vectorised_values_match_the_registered_metrics_directly(self, pairs):
        labels, predictions, response = pairs

        groups, scores = grouped_scores(labels, predictions, response)

        mask = labels == groups[2]
        for metric, function in AVAILABLE_METRICS.items():
            assert scores[metric][2] == pytest.approx(function(y_pred=predictions[mask], y_true=response[mask]))

    def test_all_metrics_are_computed_by_default(self, pairs):
        _, scores = grouped_scores(*pairs)

        assert set(scores) == set(AVAILABLE_METRICS)

    def test_groups_are_the_sorted_distinct_labels(self, pairs):
        groups, scores = grouped_scores(*pairs, ["MSE"])

        assert list(groups) == [f"D{i}" for i in range(6)]
        assert scores["MSE"].shape == (6,)

    def test_non_finite_pairs_are_dropped_before_grouping(self):
        labels = np.array(["a", "a", "a", "b"])
        predictions = np.array([1.0, 2.0, np.nan, 5.0])
        response = np.array([1.0, 2.0, 3.0, np.nan])

        groups, scores = grouped_scores(labels, predictions, response, ["Pearson", "MSE"])

        assert list(groups) == ["a"]
        assert scores["Pearson"][0] == pytest.approx(1.0)

    def test_an_unknown_metric_is_rejected(self, pairs):
        with pytest.raises(ValueError, match="invalid metric Nope"):
            grouped_scores(*pairs, ["Nope"])

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="equally long"):
            grouped_scores(np.array(["a", "b"]), np.array([1.0]), np.array([1.0, 2.0]))


class TestUndefinedGroups:
    """A group a metric cannot score is NaN - never 0 and never 1."""

    labels = np.repeat(["const_pred", "const_true", "varying", "single"], [4, 4, 4, 1])
    predictions = np.array([1.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 4.0, 3.0] + [7.0])
    response = np.array([1.0, 2.0, 3.0, 4.0] + [2.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [7.0])

    @pytest.fixture(scope="class")
    def scores(self) -> dict[str, np.ndarray]:
        groups, scores = grouped_scores(self.labels, self.predictions, self.response)
        assert list(groups) == ["const_pred", "const_true", "single", "varying"]
        return scores

    @pytest.mark.parametrize("metric", ["Pearson", "Spearman", "Kendall"])
    def test_a_constant_prediction_has_no_correlation(self, scores, metric):
        assert np.isnan(scores[metric][0])

    @pytest.mark.parametrize("metric", ["Pearson", "Spearman", "Kendall", "R^2"])
    def test_a_constant_response_has_no_correlation_or_r2(self, scores, metric):
        assert np.isnan(scores[metric][1])

    @pytest.mark.parametrize("metric", ["Pearson", "Spearman", "Kendall", "R^2"])
    def test_a_single_pair_group_is_undefined(self, scores, metric):
        assert np.isnan(scores[metric][2])

    @pytest.mark.parametrize("metric", ["MSE", "RMSE", "MAE"])
    def test_error_metrics_are_defined_for_every_group(self, scores, metric):
        assert np.isfinite(scores[metric]).all()

    def test_a_constant_prediction_still_has_an_r2(self, scores):
        assert np.isfinite(scores["R^2"][0])

    def test_a_varying_group_is_scored(self, scores):
        assert scores["Pearson"][3] == pytest.approx(0.8)

    def test_a_nearly_constant_prediction_counts_as_constant(self):
        predictions = 5.0 + np.array([0.0, CONSTANT_TOLERANCE / 10, 0.0, CONSTANT_TOLERANCE / 10])

        _, scores = grouped_scores(np.zeros(4, dtype=int), predictions, np.arange(4.0), ["Pearson", "Spearman"])

        assert np.isnan(scores["Pearson"][0])
        assert np.isnan(scores["Spearman"][0])

    def test_a_large_constant_is_not_mistaken_for_variation(self):
        predictions = np.full(50, 1234.5678901)

        _, scores = grouped_scores(np.zeros(50, dtype=int), predictions, np.arange(50.0), ["Pearson"])

        assert np.isnan(scores["Pearson"][0])


class TestMeanOverGroups:
    def test_undefined_groups_are_skipped_not_counted_as_zero(self):
        assert mean_over_groups(np.array([0.5, np.nan, 0.3])) == (pytest.approx(0.4), 2)

    def test_no_defined_group_gives_nan_and_zero(self):
        mean, count = mean_over_groups(np.array([np.nan, np.nan]))

        assert np.isnan(mean)
        assert count == 0

    def test_an_empty_array_gives_nan_and_zero(self):
        mean, count = mean_over_groups(np.array([]))

        assert np.isnan(mean)
        assert count == 0

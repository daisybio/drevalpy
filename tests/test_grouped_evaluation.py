"""Tests for :mod:`drevalpy._grouped_evaluation`: a group a metric cannot score is ``NaN``, never ``0`` or ``1``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy._grouped_evaluation import (
    GROUPINGS,
    _group_score,
    defined_labels,
    group_labels,
    grouped_scores,
    mean_over_groups,
    run_group_score,
)
from drevalpy.evaluation import AVAILABLE_METRICS, CONSTANT_ATOL
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
    def test_drug_sample_and_study_groupings(self):
        run = make_run_result(n_pairs=4)
        run.sample_ids = np.array(["GDSC2::CL_1", "CTRPv2::CL_1", "GDSC2::a::b", "CTRPv2::CL_2"])

        assert GROUPINGS == ("drug", "sample", "study")
        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)
        np.testing.assert_array_equal(group_labels(run, "sample"), run.sample_ids)
        assert group_labels(run, "study").tolist() == ["GDSC2", "CTRPv2", "GDSC2", "CTRPv2"]
        assert defined_labels(group_labels(run, "drug")).all()

    @pytest.mark.parametrize("sample_id", ["CL_1", "::CL_1"])
    def test_a_sample_id_without_a_study_prefix_has_no_study(self, sample_id):
        run = make_run_result(n_pairs=2)
        run.sample_ids = np.array(["GDSC2::CL_1", sample_id])

        labels = group_labels(run, "study")

        assert labels[0] == "GDSC2"
        assert labels[1] is np.nan
        assert defined_labels(labels).tolist() == [True, False]

    def test_an_unknown_grouping_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            group_labels(make_run_result(), "tissue")


class TestGroupedScores:
    def test_pairs_without_a_group_or_with_non_finite_values_are_dropped(self):
        labels = np.array(["A", "A", np.nan, "B", "B", np.nan, "B"], dtype=object)
        predictions = np.array([1.0, 2.0, 100.0, 1.0, 3.0, -100.0, np.nan])
        response = np.array([1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 4.0])

        groups, scores = grouped_scores(labels, predictions, response, ["MSE"])

        assert groups.tolist() == ["A", "B"]
        np.testing.assert_allclose(scores["MSE"], [0.0, 0.5])

    def test_labels_that_are_all_undefined_leave_no_groups(self):
        labels = np.array([np.nan, np.nan], dtype=object)

        groups, scores = grouped_scores(labels, np.array([1.0, 2.0]), np.array([1.0, 3.0]), ["MSE"])

        assert groups.size == scores["MSE"].size == 0

    def test_every_metric_matches_the_registered_function_per_group(self, pairs):
        labels, predictions, response = pairs

        groups, scores = grouped_scores(labels, predictions, response)

        assert set(scores) == set(AVAILABLE_METRICS)
        assert list(groups) == [f"D{i}" for i in range(6)]
        for metric, function in AVAILABLE_METRICS.items():
            for index, group in enumerate(groups):
                mask = labels == group
                assert scores[metric][index] == pytest.approx(
                    _group_score(metric, predictions[mask], response[mask]), abs=1e-9
                )
                assert scores[metric][index] == pytest.approx(function(y_pred=predictions[mask], y_true=response[mask]))

    def test_an_unknown_metric_and_mismatched_lengths_are_rejected(self, pairs):
        with pytest.raises(ValueError, match="invalid metric Nope"):
            grouped_scores(*pairs, ["Nope"])
        with pytest.raises(ValueError, match="equally long"):
            grouped_scores(np.array(["a", "b"]), np.array([1.0]), np.array([1.0, 2.0]))


class TestUndefinedGroups:
    labels = np.repeat(["const_pred", "const_true", "varying", "single"], [4, 4, 4, 1])
    predictions = np.array([1.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [1.0, 2.0, 4.0, 3.0] + [7.0])
    response = np.array([1.0, 2.0, 3.0, 4.0] + [2.0] * 4 + [1.0, 2.0, 3.0, 4.0] + [7.0])

    @pytest.fixture(scope="class")
    def scores(self) -> dict[str, np.ndarray]:
        groups, scores = grouped_scores(self.labels, self.predictions, self.response)
        assert list(groups) == ["const_pred", "const_true", "single", "varying"]
        return scores

    @pytest.mark.parametrize(
        ("group", "metrics"),
        [
            pytest.param(0, ["Pearson", "Spearman", "Kendall"], id="constant-prediction"),
            pytest.param(1, ["Pearson", "Spearman", "Kendall", "R^2"], id="constant-response"),
            pytest.param(2, ["Pearson", "Spearman", "Kendall", "R^2"], id="single-pair"),
        ],
    )
    def test_the_group_is_undefined(self, scores, group, metrics):
        assert all(np.isnan(scores[metric][group]) for metric in metrics)

    def test_error_metrics_and_a_varying_group_are_scored(self, scores):
        assert all(np.isfinite(scores[metric]).all() for metric in ("MSE", "RMSE", "MAE"))
        assert np.isfinite(scores["R^2"][0])
        assert scores["Pearson"][3] == pytest.approx(0.8)

    @pytest.mark.parametrize(
        "predictions",
        [
            pytest.param(5.0 + np.array([0.0, CONSTANT_ATOL / 10, 0.0, CONSTANT_ATOL / 10]), id="nearly-constant"),
            pytest.param(np.full(4, 1234.5678901), id="large-constant"),
        ],
    )
    def test_a_constant_prediction_is_not_mistaken_for_variation(self, predictions):
        _, scores = grouped_scores(np.zeros(4, dtype=int), predictions, np.arange(4.0), ["Pearson", "Spearman"])

        assert np.isnan(scores["Pearson"][0])
        assert np.isnan(scores["Spearman"][0])


class TestMeanOverGroups:
    def test_undefined_groups_are_skipped_not_counted_as_zero(self):
        assert mean_over_groups(np.array([0.5, np.nan, 0.3])) == (pytest.approx(0.4), 2)

    @pytest.mark.parametrize("values", [np.array([np.nan, np.nan]), np.array([])])
    def test_no_defined_group_gives_nan_and_zero(self, values):
        mean, count = mean_over_groups(values)

        assert np.isnan(mean)
        assert count == 0


class TestRunGroupScore:
    def test_averages_the_defined_drugs_of_a_run(self):
        run = make_run_result(n_pairs=20, n_drugs=4)
        per_drug = [
            np.corrcoef(run.predictions[run.drug_ids == d], run.ground_truth[run.drug_ids == d])[0, 1]
            for d in np.unique(run.drug_ids)
        ]

        assert run_group_score(run, "drug", "Pearson") == pytest.approx(np.mean(per_drug))

    def test_a_run_without_a_defined_group_is_nan_and_an_unknown_grouping_is_rejected(self):
        assert np.isnan(run_group_score(make_run_result(n_pairs=1, n_drugs=1, n_samples=1), "drug", "Pearson"))
        with pytest.raises(ValueError, match="Unknown grouping"):
            run_group_score(make_run_result(), "tissue", "Pearson")  # type: ignore[arg-type]

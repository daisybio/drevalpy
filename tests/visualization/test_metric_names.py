"""Tests for :mod:`drevalpy.visualization._metric_names`.

These functions encode the metric-naming contract between
:meth:`~drevalpy.types.results.experiment.ExperimentResult.normalize`, which
recomputes metrics under their plain names, and the plots that consume them.
Getting it wrong produced an all-NaN leaderboard column and a crash in
``set_xlim``, so the resolution order is pinned here rather than only through the
plots.
"""

from __future__ import annotations

import pytest

from drevalpy.visualization._metric_names import (
    NORMALIZED_SUFFIX,
    holds_normalized_values,
    metric_keys,
    resolve_metric_key,
)
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_model_result, make_run_result


class TestNormalizedSuffix:
    def test_matches_the_legacy_column_spelling(self):
        assert f"Pearson{NORMALIZED_SUFFIX}" == "Pearson: normalized"


class TestMetricKeys:
    def test_collects_the_keys_of_an_experiment(self):
        experiment = make_experiment_result(n_models=2, n_folds=2)

        assert "Pearson" in metric_keys(experiment)

    def test_accepts_a_model_result(self):
        model = make_model_result(n_folds=2)

        assert metric_keys(model) == set(model.runs[0].metrics)

    def test_unions_keys_that_differ_between_runs(self):
        from drevalpy.types.results.experiment import ExperimentResult

        result = ExperimentResult(
            [
                make_run_result(model_name="A", metrics={"MSE": 1.0}),
                make_run_result(model_name="B", metrics={"Pearson": 0.5}),
            ]
        )

        assert metric_keys(result) == {"MSE", "Pearson"}


class TestResolveMetricKey:
    def test_the_plain_name_is_returned_when_present(self):
        assert resolve_metric_key({"Pearson", "Pearson: normalized"}, "Pearson") == "Pearson"

    def test_falls_back_to_the_suffixed_name(self):
        assert resolve_metric_key({"Pearson: normalized"}, "Pearson") == "Pearson: normalized"

    def test_returns_none_when_the_metric_is_absent(self):
        assert resolve_metric_key({"MSE"}, "Pearson") is None

    def test_accepts_any_iterable_of_names(self):
        assert resolve_metric_key(iter(["RMSE"]), "RMSE") == "RMSE"

    @pytest.mark.parametrize("base", ["MSE", "RMSE", "MAE", "R^2", "Pearson", "Spearman", "Kendall"])
    def test_every_reported_metric_resolves_on_a_normalized_experiment(self, base):
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL)

        assert resolve_metric_key(metric_keys(normalized), base) == base


class TestHoldsNormalizedValues:
    def test_true_for_a_normalized_experiment_under_the_plain_name(self):
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize(REFERENCE_MODEL)

        assert holds_normalized_values(normalized, "Pearson") is True

    def test_true_for_the_suffixed_key_regardless_of_the_container(self):
        experiment = make_experiment_result(n_models=2, n_folds=2)

        assert holds_normalized_values(experiment, "Pearson: normalized") is True

    def test_false_for_a_plain_key_on_an_unnormalized_experiment(self):
        experiment = make_experiment_result(n_models=2, n_folds=2)

        assert holds_normalized_values(experiment, "Pearson") is False

    def test_false_for_a_model_result_which_tracks_no_reference(self):
        assert holds_normalized_values(make_model_result(n_folds=2), "Pearson") is False

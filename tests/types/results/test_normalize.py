"""Tests for normalizing one run against a reference run."""

from __future__ import annotations

import numpy as np

from drevalpy.types.results._normalize import normalize_run, reference_summary
from tests.synthetic import make_run_result


class TestNormalizeRun:
    def test_subtracts_the_reference_prediction_per_pair(self) -> None:
        run = make_run_result(model_name="A", seed=1)
        reference = make_run_result(model_name="Ref", seed=2)

        normalized = normalize_run(run, reference)

        np.testing.assert_allclose(normalized.predictions, run.predictions - reference.predictions)
        np.testing.assert_allclose(normalized.ground_truth, run.ground_truth - reference.predictions)

    def test_keeps_the_unnormalized_metrics(self) -> None:
        run = make_run_result(model_name="A", seed=1)

        normalized = normalize_run(run, make_run_result(model_name="Ref", seed=2))

        assert normalized.raw_metrics == run.metrics

    def test_pairs_the_reference_never_predicted_normalize_against_zero(self) -> None:
        run = make_run_result(model_name="A", n_pairs=20, seed=1)
        reference = make_run_result(model_name="Ref", n_pairs=10, seed=2)

        normalized = normalize_run(run, reference)

        np.testing.assert_allclose(normalized.predictions[10:], run.predictions[10:])


class TestReferenceSummary:
    def test_drops_the_arrays_and_keeps_the_metrics(self) -> None:
        reference = make_run_result(model_name="Ref", seed=2)

        summary = reference_summary(reference)

        assert summary.predictions.size == 0
        assert summary.sample_ids.size == 0
        assert summary.raw_metrics == reference.metrics

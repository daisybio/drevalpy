"""Normalization of runs against a reference model's runs."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.evaluation import Metric
from drevalpy.evaluation._scoring import run_per_group_score
from drevalpy.types.results._normalize import normalize_run, reference_summary
from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.run import _intern_ids
from drevalpy.types.results.trial import TrialResult
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_result


def _trial(n: int = 4) -> TrialResult:
    return TrialResult(
        hyperparameters={"alpha": 0.1},
        metrics={"MSE": 0.3},
        optimization_metric="MSE",
        predictions=np.zeros(n),
    )


def _normalized_run(reference, other):
    return ExperimentResult([reference, other]).normalize().models[0].runs[0]


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


class TestNormalize:
    def test_subtracts_the_reference_and_drops_it(self) -> None:
        experiment = make_experiment_result(n_models=3, n_folds=2, split_mode="LSO")
        reference, other = experiment.models[:2]

        normalized = experiment.normalize()
        run = normalized.models[0].runs[1]

        assert REFERENCE_MODEL not in normalized.model_names
        assert (normalized.n_models, normalized.normalized_by) == (2, REFERENCE_MODEL)
        np.testing.assert_allclose(run.predictions, other.runs[1].predictions - reference.runs[1].predictions)
        np.testing.assert_allclose(run.ground_truth, other.runs[1].ground_truth - reference.runs[1].predictions)
        assert (run.fold_id, run.fold_index, run.split_mode) == ("fold_1", 1, "LSO")
        assert set(run.metrics) == set(Metric)

    def test_keeps_the_unnormalized_metrics_of_every_run(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)
        other = experiment.models[1].runs[0]

        run = experiment.normalize().models[0].runs[0]

        assert run.raw_metrics == other.metrics
        assert run.metrics != other.metrics
        assert run.raw_per_drug_pearson == pytest.approx(run_per_group_score(other, "drug", "Pearson"))

    def test_summarizes_the_reference_runs_without_their_arrays(self) -> None:
        experiment = make_experiment_result(n_models=3, n_folds=2)
        normalized = experiment.normalize()

        assert [run.model_name for run in normalized.reference_runs] == [REFERENCE_MODEL] * 2
        summary = normalized.reference_runs[0]
        assert summary.raw_metrics == experiment.models[0].runs[0].metrics
        assert summary.metrics["Pearson"] == 0.0
        assert (summary.predictions.size, summary.sample_ids.size, summary.trials) == (0, 0, None)
        assert experiment.reference_runs == ()

    def test_treats_pairs_missing_from_the_reference_as_zero(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=4, n_samples=2, n_drugs=2)
        other = make_run_result(model_name="ElasticNet", n_pairs=4, n_samples=2, n_drugs=2)
        other.sample_ids = np.array(["CL_0", "CL_9", "CL_0", "CL_9"])
        other.drug_ids = np.array(["D_0", "D_0", "D_1", "D_1"])

        run = _normalized_run(reference, other)

        ref_by_pair = dict(
            zip(zip(reference.sample_ids, reference.drug_ids, strict=True), reference.predictions, strict=True)
        )
        offsets = [ref_by_pair.get(pair, 0.0) for pair in zip(other.sample_ids, other.drug_ids, strict=True)]
        np.testing.assert_allclose(run.predictions, other.predictions - np.array(offsets))

    def test_yields_no_metrics_when_every_pair_is_nan(self) -> None:
        other = make_run_result(model_name="ElasticNet", n_pairs=4)
        other.predictions = np.full(4, np.nan)

        assert _normalized_run(make_run_result(model_name=REFERENCE_MODEL, n_pairs=4), other).metrics == {}

    def test_carries_trials_over(self) -> None:
        other = make_run_result(model_name="ElasticNet", n_pairs=4)
        other.trials = [_trial()]

        run = _normalized_run(make_run_result(model_name=REFERENCE_MODEL, n_pairs=4), other)

        assert run.trials is not None
        assert run.trials[0].hyperparameters == {"alpha": 0.1}

    def test_rejects_a_second_normalization(self) -> None:
        with pytest.raises(ValueError, match="Already normalized"):
            make_experiment_result(n_models=2, n_folds=1).normalize().normalize()

    def test_rejects_an_unknown_reference_model(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            make_experiment_result(n_models=2, n_folds=1).normalize(reference_model="NotAModel")

    def test_rejects_a_fold_without_a_reference_run(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, fold_index=0)
        other = make_run_result(model_name="ElasticNet", fold_index=1)

        with pytest.raises(ValueError, match="No reference run"):
            ExperimentResult([reference, other]).normalize()

    def test_accepts_an_explicit_reference_model(self) -> None:
        normalized = make_experiment_result(n_models=3, n_folds=1).normalize(reference_model="ElasticNet")

        assert normalized.normalized_by == "ElasticNet"
        assert "ElasticNet" not in normalized.model_names

    def test_randomized_variants_normalize_against_the_ordinary_reference(self) -> None:
        ordinary = make_run_result(model_name=REFERENCE_MODEL, fold_index=0)
        randomized_reference = make_run_result(
            model_name=REFERENCE_MODEL, fold_index=0, randomization=("SVRC", "gene_expression")
        )
        other = make_run_result(model_name="ElasticNet", fold_index=0)
        randomized = make_run_result(model_name="ElasticNet", fold_index=0, randomization=("SVRC", "gene_expression"))

        normalized = ExperimentResult([randomized_reference, other, ordinary]).normalize()
        np.testing.assert_allclose(normalized.models[0].runs[0].predictions, other.predictions - ordinary.predictions)

        run = _normalized_run(ordinary, randomized)
        np.testing.assert_allclose(run.predictions, randomized.predictions - ordinary.predictions)

    def test_normalization_is_independent_of_reference_run_order(self) -> None:
        runs = {
            (name, fold): make_run_result(model_name=name, fold_index=fold)
            for name in (REFERENCE_MODEL, "ElasticNet")
            for fold in (0, 1)
        }
        keys = list(runs)

        first = ExperimentResult([runs[k] for k in keys]).normalize()
        second = ExperimentResult([runs[k] for k in reversed(keys)]).normalize()

        assert {run.fold_key: run.run_id for run in first.models[0].runs} == {
            run.fold_key: run.run_id for run in second.models[0].runs
        }

    def test_rejects_duplicate_ordinary_reference_runs_for_one_fold(self) -> None:
        runs = [
            make_run_result(model_name=REFERENCE_MODEL, fold_index=0),
            make_run_result(model_name=REFERENCE_MODEL, fold_index=0),
            make_run_result(model_name="ElasticNet", fold_index=0),
        ]

        with pytest.raises(ValueError, match="Duplicate ordinary reference"):
            ExperimentResult(runs).normalize()

    def test_normalizes_interned_id_arrays(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=8)
        other = make_run_result(model_name="ElasticNet", n_pairs=8)
        for run in (reference, other):
            run.sample_ids = _intern_ids(run.sample_ids)
            run.drug_ids = _intern_ids(run.drug_ids)

        np.testing.assert_allclose(
            _normalized_run(reference, other).predictions, other.predictions - reference.predictions
        )

    def test_a_duplicated_reference_pair_resolves_to_its_last_prediction(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=2)
        reference.sample_ids = np.array(["CL_0", "CL_0"])
        reference.drug_ids = np.array(["D_0", "D_0"])
        reference.predictions = np.array([1.0, 5.0])
        other = make_run_result(model_name="ElasticNet", n_pairs=1)
        other.sample_ids = np.array(["CL_0"])
        other.drug_ids = np.array(["D_0"])
        other.predictions = other.ground_truth = np.array([7.0])

        np.testing.assert_allclose(_normalized_run(reference, other).predictions, np.array([2.0]))

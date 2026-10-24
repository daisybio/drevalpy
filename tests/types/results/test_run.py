"""Per-fold run results: persistence, trials, identity, normalization and failure records."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import numpy as np
import pytest
from upath import UPath

from drevalpy.errors import DrEvalError
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results._normalize import normalize_run, reference_summary
from drevalpy.types.results.failure import ExperimentFailedError, RunFailure
from drevalpy.types.results.run import RunResult, intern_ids
from drevalpy.types.results.trial import TrialResult
from tests.synthetic import make_metrics, make_run_result


def _trial(seed: int) -> TrialResult:
    return TrialResult(
        hyperparameters={"alpha": 0.1 * seed},
        metrics={"MSE": 0.5 - 0.1 * seed},
        optimization_metric="MSE",
        predictions=np.full(4, float(seed)),
    )


def _roundtrip(run: RunResult, tmp_path, **load_kwargs) -> RunResult:
    path = UPath(tmp_path) / "fold.npz"
    run.save(path)
    return RunResult.load(path, **load_kwargs)


class TestIdentity:
    def test_optional_metadata_defaults_to_empty(self) -> None:
        run = RunResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            fold_index=0,
            predictions=np.zeros(3),
            ground_truth=np.zeros(3),
            sample_ids=np.array(["CL_0", "CL_1", "CL_2"]),
            drug_ids=np.array(["D_0", "D_1", "D_2"]),
        )

        assert (run.split_mode, run.fold_id, run.best_hyperparameters, run.metrics) == ("", "", {}, {})
        assert run.fold_metadata == {}
        assert run.trials is None
        assert run.randomization is None
        text = repr(run)
        assert "Hyperparameters:" not in text
        assert "Metrics:" not in text

    def test_builder_derives_fold_id_and_metrics(self) -> None:
        assert make_run_result(fold_index=2).fold_id == "fold_2"
        assert make_run_result().metrics.keys() == make_metrics().keys()

    def test_fold_key_prefers_the_stable_fold_id_else_the_index(self) -> None:
        assert make_run_result(fold_index=7, fold_id="stable").fold_key == ("SyntheticDataset", "LPO", "id", "stable")
        assert make_run_result(fold_index=7, fold_id="").fold_key == ("SyntheticDataset", "LPO", "index", 7)

    def test_execution_id_distinguishes_diagnostic_variants(self) -> None:
        ordinary = make_run_result(fold_index=0)
        randomized = make_run_result(fold_index=0, randomization=("SVRC", "gene_expression"))
        robustness = make_run_result(fold_index=0, fold_metadata={"robustness_trial": 0})

        assert ordinary.execution_id != randomized.execution_id
        assert ordinary.execution_id != robustness.execution_id


class TestRepr:
    def test_reports_the_run_summary(self) -> None:
        run = make_run_result(
            model_name="RandomForest",
            n_pairs=5,
            fold_index=1,
            fold_metadata={"fold_index": 1, "robustness_trial": 3},
            best_hyperparameters={"alpha": 0.25},
            metrics={"MSE": 0.5},
        )
        run.ground_truth = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        run.trials = [_trial(0), _trial(1)]

        text = repr(run)
        lines = text.splitlines()

        assert text.startswith("RunResult")
        for expected in (
            "Model: RandomForest",
            "Dataset: SyntheticDataset",
            "Randomization: None",
            "Predictions: 5 pairs",
            "Ground truth: 3 non-NaN values",
            "HPO Trials: 2",
        ):
            assert expected in text
        assert "        robustness_trial: 3" in lines
        assert "        fold_index: 1" not in lines
        assert "    Hyperparameters:" in lines
        assert "        alpha: 0.25" in lines
        assert "        MSE: 0.5000" in lines

    def test_reports_randomization_and_omits_absent_trials(self) -> None:
        run = make_run_result(randomization=("SVRC", "gene_expression"))

        assert "Randomization: SVRC (gene_expression)" in repr(run)
        assert "HPO Trials" not in repr(run)


class TestPersistence:
    def test_round_trip_preserves_arrays_and_metadata(self, tmp_path) -> None:
        run = make_run_result(
            n_pairs=9,
            model_name="RandomForest",
            fold_index=2,
            split_mode="LSO",
            best_hyperparameters={"alpha": 0.25},
            fold_metadata={"note": "kept"},
            randomization=("SVRC", "gene_expression"),
        )
        run.raw_metrics = {"RMSE": 1.5, "Pearson": 0.75}
        run.raw_per_drug_pearson = 0.25

        loaded = _roundtrip(run, tmp_path)

        np.testing.assert_allclose(loaded.predictions, run.predictions)
        np.testing.assert_allclose(loaded.ground_truth, run.ground_truth)
        np.testing.assert_array_equal(loaded.sample_ids, run.sample_ids)
        np.testing.assert_array_equal(loaded.drug_ids, run.drug_ids)
        assert (loaded.model_name, loaded.dataset_name) == ("RandomForest", run.dataset_name)
        assert (loaded.split_mode, loaded.fold_index, loaded.fold_id) == ("LSO", 2, "fold_2")
        assert loaded.metrics == pytest.approx(run.metrics)
        assert loaded.best_hyperparameters == {"alpha": 0.25}
        assert loaded.fold_metadata == {"note": "kept"}
        assert loaded.raw_metrics == {"RMSE": 1.5, "Pearson": 0.75}
        assert loaded.raw_per_drug_pearson == 0.25
        assert loaded.randomization == ("SVRC", "gene_expression")

    def test_optional_fields_default_when_absent(self, tmp_path) -> None:
        loaded = _roundtrip(make_run_result(), tmp_path)

        assert loaded.raw_metrics == {}
        assert loaded.raw_per_drug_pearson is None
        assert loaded.randomization is None
        assert loaded.trials is None

    def test_results_recording_a_removed_execution_flag_still_load(self, tmp_path) -> None:
        removed_flag = "precomputed" + "_only"
        run = make_run_result(fold_metadata={"execution": {"hpo": {"enabled": False}, removed_flag: False}})

        loaded = _roundtrip(run, tmp_path)

        assert loaded.fold_metadata["execution"][removed_flag] is False
        assert loaded.identity.execution_key == run.identity.execution_key

    def test_trials_round_trip_and_can_be_skipped(self, tmp_path) -> None:
        run = make_run_result(n_pairs=6)
        run.trials = [_trial(0), _trial(1)]

        loaded = _roundtrip(run, tmp_path)

        assert loaded.trials is not None
        assert [t.hyperparameters for t in loaded.trials] == [t.hyperparameters for t in run.trials]
        assert [t.optimization_metric for t in loaded.trials] == ["MSE", "MSE"]
        np.testing.assert_allclose(loaded.trials[1].predictions, np.full(4, 1.0))

        skipped = RunResult.load(UPath(tmp_path) / "fold.npz", with_trials=False)
        assert skipped.trials is None
        np.testing.assert_allclose(skipped.predictions, run.predictions)
        np.testing.assert_array_equal(skipped.drug_ids, run.drug_ids)

    def test_empty_trial_list_is_not_serialized(self, tmp_path) -> None:
        run = make_run_result(split_mode="LDO")
        run.trials = []
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["_metadata"]))
        assert meta["trials"] is None
        assert meta["split_mode"] == "LDO"
        assert meta["fold_id"] == "fold_0"

    def test_accepts_a_plain_string_path(self, tmp_path) -> None:
        path = str(UPath(tmp_path) / "fold.npz")
        make_run_result(fold_index=1).save(path)

        assert RunResult.load(path).fold_index == 1

    def test_entity_ids_are_stored_as_strings(self, tmp_path) -> None:
        run = make_run_result(n_pairs=3)
        run.sample_ids = np.array([1, 2, 3])

        np.testing.assert_array_equal(_roundtrip(run, tmp_path).sample_ids, np.array(["1", "2", "3"]))

    def test_a_file_written_before_the_sample_rename_is_rejected(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result().save(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {("cell_line_ids" if key == "sample_ids" else key): data[key] for key in data.files}
        np.savez_compressed(path, **arrays)

        with pytest.raises(ValueError, match="written before drevalpy renamed cell lines to samples; re-run"):
            RunResult.load(path)


class TestInternIds:
    def test_values_are_preserved_as_shared_str_objects(self) -> None:
        ids = np.array(["CL_0", "CL_1", "CL_0"] + ["CL_0"] * 5, dtype="<U40")

        interned = intern_ids(ids)

        np.testing.assert_array_equal(interned, ids)
        assert interned.dtype == object
        assert isinstance(interned[0], str)
        assert len({id(value) for value in interned.tolist() if value == "CL_0"}) == 1

    def test_empty_and_pointer_table_size(self) -> None:
        assert intern_ids(np.array([], dtype="<U40")).shape == (0,)
        ids = np.array([f"BRD-K{i:035d}" for i in range(4)] * 250, dtype="<U40")
        assert intern_ids(ids).nbytes < ids.nbytes

    def test_load_interns_and_stays_lossless_across_resaves(self, tmp_path) -> None:
        run = make_run_result(n_pairs=8, n_samples=3, n_drugs=2)
        first = UPath(tmp_path) / "a.npz"
        run.save(first)
        once = RunResult.load(first)
        once.save(UPath(tmp_path) / "b.npz")
        twice = RunResult.load(UPath(tmp_path) / "b.npz")

        assert once.sample_ids.dtype == object
        assert once.drug_ids.dtype == object
        np.testing.assert_array_equal(once.sample_ids, run.sample_ids)
        np.testing.assert_array_equal(twice.drug_ids, run.drug_ids)


def _trial_result(**overrides: Any) -> TrialResult:
    kwargs: dict[str, Any] = {
        "hyperparameters": {"alpha": 0.5},
        "metrics": {"MSE": 0.25, "Pearson": 0.75},
        "optimization_metric": "MSE",
        "predictions": np.array([1.0, 2.0, 3.0]),
    }
    kwargs.update(overrides)
    return TrialResult(**kwargs)


def test_score_returns_value_of_optimization_metric() -> None:
    trial = _trial_result(optimization_metric="Pearson")

    assert trial.score == 0.75


def test_score_falls_back_to_nan_when_metric_missing() -> None:
    trial = _trial_result(optimization_metric="Kendall")

    assert math.isnan(trial.score)


def test_score_is_nan_for_empty_metrics() -> None:
    trial = _trial_result(metrics={})

    assert math.isnan(trial.score)


def test_repr_lists_every_hyperparameter() -> None:
    trial = _trial_result(hyperparameters={"alpha": 0.5, "l1_ratio": 0.1})

    text = repr(trial)

    assert "alpha: 0.5" in text
    assert "l1_ratio: 0.1" in text


def test_repr_marks_only_the_optimization_metric() -> None:
    trial = _trial_result(optimization_metric="MSE")

    lines = repr(trial).splitlines()

    assert "        MSE: 0.2500 *" in lines
    assert "        Pearson: 0.7500" in lines


def test_repr_starts_with_the_type_name() -> None:
    assert repr(_trial_result()).startswith("TrialResult")


def _identity(metadata=None, randomization=None):
    return RunIdentity.from_metadata("model", "data", "LPO", 2, "fold", randomization, metadata or {})


def test_identity_is_independent_of_metadata_key_order():
    first = _identity({"randomization_provenance": {"type": "permutation", "effective_seed": 7}})
    second = _identity({"randomization_provenance": {"effective_seed": 7, "type": "permutation"}})
    assert first == second


def test_variants_have_distinct_execution_keys_and_one_fold_key():
    ordinary = _identity()
    variants = [
        ordinary,
        _identity({"robustness_trial": 0}),
        _identity({"randomization_provenance": {"effective_seed": 7}}, ("SVRC", "expression")),
        _identity({"randomization_provenance": {"effective_seed": 8}}, ("SVRC", "expression")),
    ]
    assert len({v.execution_key for v in variants}) == len(variants)
    assert {v.fold_key for v in variants} == {ordinary.fold_key}


def test_legacy_fold_index_cannot_collide_with_a_string_fold_id():
    old = RunIdentity("m", "d", "LPO", 2)
    new = RunIdentity("m", "d", "LPO", 2, "2")
    assert old.fold_key != new.fold_key


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


def _failure(**overrides) -> RunFailure:
    fields = {
        "model_name": "A",
        "fold_index": 1,
        "randomization": None,
        "robustness_trial": None,
        "error_type": "ValueError",
        "message": "boom",
    }
    return RunFailure(**{**fields, **overrides})


def test_from_exception_records_type_message_and_stress_tests() -> None:
    failure = RunFailure.from_exception(KeyError("gene"), model_name="A", fold_index=2)
    assert failure == _failure(fold_index=2, error_type="KeyError", message="'gene'")

    failure = RunFailure.from_exception(
        RuntimeError("x"), model_name="A", fold_index=0, randomization=("SVRC", "gene_expression"), robustness_trial=3
    )
    assert (failure.randomization, failure.robustness_trial) == (("SVRC", "gene_expression"), 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.message = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "failure",
    [_failure(), _failure(randomization=("SVRC", "gene_expression"), robustness_trial=0)],
    ids=["plain", "stress"],
)
def test_dict_round_trips_through_json(failure: RunFailure) -> None:
    data = json.loads(json.dumps(failure.to_dict()))
    assert RunFailure.from_dict(data) == failure


def test_dict_edge_cases() -> None:
    assert _failure(randomization=("SVRC", "gene_expression")).to_dict()["randomization"] == ["SVRC", "gene_expression"]
    data = {"model_name": "A", "fold_index": 1, "error_type": "ValueError", "message": "boom"}
    assert RunFailure.from_dict(data) == _failure()


def test_str_names_model_fold_error_and_stress_tests() -> None:
    assert str(_failure()) == "A fold 1: ValueError: boom"
    text = str(_failure(randomization=("SVRC", "gene_expression"), robustness_trial=2))
    assert "randomization SVRC (gene_expression)" in text
    assert "robustness trial 2" in text


def test_experiment_failed_error_carries_and_lists_the_failures() -> None:
    failures = [_failure(fold_index=i) for i in range(8)]
    error = ExperimentFailedError(failures)

    assert isinstance(error, RuntimeError)
    assert isinstance(error, DrEvalError)
    assert error.failures == tuple(failures)
    lines = str(error).splitlines()
    assert lines[0] == "8 run(s) failed:"
    assert lines[1:] == [f"  {failure}" for failure in failures]

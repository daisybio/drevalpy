"""Experiment-level result: guards, capabilities, persistence and load logging."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from upath import UPath

from drevalpy.errors import InvalidInputError
from drevalpy.types.results.experiment import ExperimentResult, _run_array_bytes
from drevalpy.types.results.run import _intern_ids
from drevalpy.types.results.trial import TrialResult
from tests.synthetic import REFERENCE_MODEL, make_experiment_result, make_run_failure, make_run_result


def _trial(n: int = 4) -> TrialResult:
    return TrialResult(
        hyperparameters={"alpha": 0.1},
        metrics={"MSE": 0.3},
        optimization_metric="MSE",
        predictions=np.zeros(n),
    )


class TestConstruction:
    def test_rejects_an_empty_run_list(self) -> None:
        with pytest.raises(InvalidInputError, match="must not be empty"):
            ExperimentResult([])

    @pytest.mark.parametrize(
        ("kwargs_a", "kwargs_b", "message"),
        [
            ({"dataset_name": "GDSC1"}, {"dataset_name": "CTRPv2"}, "same dataset_name"),
            ({"model_name": "A", "split_mode": "LPO"}, {"model_name": "B", "split_mode": "LSO"}, "same split_mode"),
        ],
    )
    def test_rejects_mixed_runs(self, kwargs_a: dict, kwargs_b: dict, message: str) -> None:
        with pytest.raises(InvalidInputError, match=message):
            ExperimentResult([make_run_result(**kwargs_a), make_run_result(**kwargs_b)])

    def test_blank_split_modes_are_ignored_by_the_consistency_check(self) -> None:
        runs = [make_run_result(model_name="A", split_mode="LPO"), make_run_result(model_name="B", split_mode="")]

        assert ExperimentResult(runs).split_mode == "LPO"
        assert ExperimentResult([make_run_result(split_mode="")]).split_mode == ""

    def test_groups_runs_by_model_name(self) -> None:
        runs = [
            make_run_result(model_name="A", fold_index=0),
            make_run_result(model_name="A", fold_index=1),
            make_run_result(model_name="B", fold_index=0),
        ]

        result = ExperimentResult(runs)

        assert result.model_names == ["A", "B"]
        assert [m.n_folds for m in result.models] == [2, 1]
        assert (result.n_models, result.max_folds, result.normalized_by) == (2, 2, None)
        assert {m.dataset_name for m in result.models} == {"SyntheticDataset"}


def test_capabilities_are_absent_by_default_and_detected_from_the_data() -> None:
    plain = make_experiment_result(n_models=2, n_folds=1)

    assert (plain.has_randomization, plain.has_robustness) == (False, False)
    assert make_experiment_result(n_models=2, n_folds=1, with_randomization=True).has_randomization is True
    assert make_experiment_result(n_models=2, n_folds=1, with_robustness=True).has_robustness is True


class TestPersistence:
    def test_save_writes_one_directory_per_model_and_only_the_metadata_the_runs_cannot_give(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"

        make_experiment_result(n_models=2, n_folds=1, split_mode="LDO").save(directory)
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        assert (directory / REFERENCE_MODEL / "fold_0.npz").is_file()
        assert (directory / "ElasticNet" / "fold_0.npz").is_file()
        assert not (directory / "ElasticNet" / "metadata.json").exists()
        assert meta == {"normalized_by": None, "models": [REFERENCE_MODEL, "ElasticNet"]}

    @pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalized"])
    def test_round_trip_preserves_structure(self, tmp_path, normalize: bool) -> None:
        experiment = make_experiment_result(n_models=3, n_folds=2)
        if normalize:
            experiment = experiment.normalize()

        experiment.save(UPath(tmp_path) / "experiment")
        loaded = ExperimentResult.load(str(UPath(tmp_path) / "experiment"))

        assert loaded.model_names == experiment.model_names
        assert loaded.max_folds == 2
        assert loaded.split_mode == experiment.split_mode
        assert loaded.normalized_by == experiment.normalized_by

    def test_spec_names_get_portable_unique_directories(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        names = ["pca[gene_expression]:fingerprints:ridge", "pca[gene_expression]_fingerprints_ridge"]
        ExperimentResult([make_run_result(model_name=name) for name in names]).save(directory)

        assert sorted(path.name for path in directory.iterdir() if path.is_dir()) == [names[1], f"{names[1]}-2"]
        assert ExperimentResult.load(directory).model_names == names

    def test_the_dataset_and_split_mode_are_those_of_the_loaded_runs(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=2, n_folds=1, dataset_name="GDSC1", split_mode="LTO").save(directory)

        loaded = ExperimentResult.load(directory)

        assert (loaded.dataset_name, loaded.split_mode) == ("GDSC1", "LTO")


class TestFailures:
    FAILURES = (
        make_run_failure("B", 1, None, None, "ValueError", "boom"),
        make_run_failure("B", 0, ("SVRC", "gene_expression"), 2, "KeyError", "'gene'"),
    )

    def test_failures_are_kept_validated_normalized_and_round_tripped(self, tmp_path) -> None:
        assert ExperimentResult([make_run_result()]).failures == ()
        assert ExperimentResult([make_run_result()], list(self.FAILURES)).failures == self.FAILURES
        with pytest.raises(InvalidInputError, match="must not be empty"):
            ExperimentResult([], self.FAILURES)

        clean = UPath(tmp_path) / "clean"
        make_experiment_result(n_models=1, n_folds=1).save(clean)
        assert not (clean / "failures.json").exists()
        assert ExperimentResult.load(clean).failures == ()

        directory = UPath(tmp_path) / "experiment"
        ExperimentResult([make_run_result()], self.FAILURES).save(directory)
        assert ExperimentResult.load(directory).failures == self.FAILURES

        experiment = make_experiment_result(n_models=2, n_folds=1)
        experiment.failures = self.FAILURES
        assert experiment.normalize().failures == self.FAILURES


class TestTrialSkipping:
    def test_trials_are_loaded_by_default_and_can_be_skipped(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        experiment = make_experiment_result(n_models=2, n_folds=1)
        for model in experiment.models:
            for run in model.runs:
                run.trials = [_trial()]
        experiment.save(directory)

        with_trials = ExperimentResult.load(directory)
        without = ExperimentResult.load(directory, with_trials=False)

        assert all(run.trials for model in with_trials.models for run in model.runs)
        assert all(run.trials is None for model in without.models for run in model.runs)
        assert without.model_names == [REFERENCE_MODEL, "ElasticNet"]


class TestLoadLogging:
    def test_reports_models_runs_rows_and_bytes(self, tmp_path, caplog) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=2, n_folds=2, n_pairs=20).save(directory)

        with caplog.at_level(logging.INFO, logger="drevalpy.types.results.experiment"):
            ExperimentResult.load(directory)
            ExperimentResult.load(directory, with_trials=False)

        loaded, skipped = (record.getMessage() for record in caplog.records[-2:])
        for fragment in ("2 models", "4 runs", "80 prediction rows", "GB of arrays", "trials loaded"):
            assert fragment in loaded
        assert "trials skipped" in skipped

    def test_byte_total_counts_trial_predictions_and_interned_strings(self) -> None:
        run = make_run_result(n_pairs=8)
        without = _run_array_bytes(run)
        run.trials = [_trial(1000)]
        assert _run_array_bytes(run) > without

        run = make_run_result(n_pairs=8)
        run.sample_ids = _intern_ids(run.sample_ids)
        run.drug_ids = _intern_ids(run.drug_ids)
        assert _run_array_bytes(run) > run.predictions.nbytes + run.ground_truth.nbytes


def test_repr_reports_metadata_and_per_model_metrics() -> None:
    assert "Split mode: LSO" in repr(make_experiment_result(n_models=2, n_folds=2, split_mode="LSO"))
    lines = repr(
        ExperimentResult(
            [make_run_result(model_name="A", metrics={"MSE": 0.5}), make_run_result(model_name="B", metrics={})]
        )
    ).splitlines()
    assert "        A (1 folds): MSE=0.5000" in lines
    assert "        B (1 folds): no metrics" in lines

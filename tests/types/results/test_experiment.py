"""Tests for the experiment-level result: guards, capabilities and normalization."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from upath import UPath

from drevalpy._grouped_evaluation import run_group_score
from drevalpy.errors import InvalidInputError
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.types.results.experiment import ExperimentResult, _run_array_bytes
from drevalpy.types.results.failure import RunFailure
from drevalpy.types.results.run import intern_ids
from drevalpy.types.results.trial import TrialResult
from drevalpy.visualization.requirements import PlotRequirement
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


class TestSatisfies:
    def test_capabilities_are_absent_by_default_and_detected_from_the_data(self) -> None:
        plain = make_experiment_result(n_models=2, n_folds=1)

        assert (plain.has_randomization, plain.has_robustness) == (False, False)
        assert make_experiment_result(n_models=2, n_folds=1, with_randomization=True).has_randomization is True
        assert make_experiment_result(n_models=2, n_folds=1, with_robustness=True).has_robustness is True

    def test_no_requirements_are_always_satisfied(self) -> None:
        assert make_experiment_result(n_models=1, n_folds=1).satisfies(frozenset()) is True

    @pytest.mark.parametrize(
        ("requirement", "n_models", "n_folds"),
        [
            pytest.param(PlotRequirement.MULTIPLE_MODELS, 1, 2, id="one-model"),
            pytest.param(PlotRequirement.MULTIPLE_FOLDS, 2, 1, id="one-fold"),
        ],
    )
    def test_structural_requirements_are_rejected_when_unmet(
        self, requirement: PlotRequirement, n_models: int, n_folds: int
    ) -> None:
        result = make_experiment_result(n_models=n_models, n_folds=n_folds)

        assert result.satisfies(frozenset({requirement})) is False

    def test_structural_requirements_are_accepted_when_met(self) -> None:
        result = make_experiment_result(n_models=2, n_folds=2)

        assert result.satisfies(frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})) is True

    @pytest.mark.parametrize(
        ("requirement", "flag"),
        [(PlotRequirement.RANDOMIZATION, "with_randomization"), (PlotRequirement.ROBUSTNESS, "with_robustness")],
    )
    def test_data_requirements_need_the_matching_data(self, requirement: PlotRequirement, flag: str) -> None:
        requirements = frozenset({requirement})

        assert make_experiment_result(n_models=2, n_folds=2).satisfies(requirements) is False
        assert make_experiment_result(n_models=2, n_folds=2, **{flag: True}).satisfies(requirements) is True


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
        assert set(run.metrics) == set(AVAILABLE_METRICS)

    def test_keeps_the_unnormalized_metrics_of_every_run(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)
        other = experiment.models[1].runs[0]

        run = experiment.normalize().models[0].runs[0]

        assert run.raw_metrics == other.metrics
        assert run.metrics != other.metrics
        assert run.raw_per_drug_pearson == pytest.approx(run_group_score(other, "drug", "Pearson"))

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

        assert {run.fold_key: run.execution_id for run in first.models[0].runs} == {
            run.fold_key: run.execution_id for run in second.models[0].runs
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
            run.sample_ids = intern_ids(run.sample_ids)
            run.drug_ids = intern_ids(run.drug_ids)

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


class TestPersistence:
    def test_save_writes_one_directory_per_model_and_experiment_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"

        make_experiment_result(n_models=2, n_folds=1, split_mode="LDO").save(directory)
        meta = json.loads((directory / "metadata.json").read_text())

        assert (directory / REFERENCE_MODEL / "metadata.json").is_file()
        assert (directory / "ElasticNet" / "metadata.json").is_file()
        assert meta["dataset_name"] == "SyntheticDataset"
        assert meta["split_mode"] == "LDO"
        assert meta["normalized_by"] is None
        assert meta["models"] == [REFERENCE_MODEL, "ElasticNet"]

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

        dirs = json.loads((directory / "metadata.json").read_text())["model_dirs"]

        assert dirs == {names[0]: names[1], names[1]: f"{names[1]}-2"}
        assert ExperimentResult.load(directory).model_names == names

    def test_load_reads_trees_saved_without_model_dirs(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=2, n_folds=1).save(directory)
        meta = json.loads((directory / "metadata.json").read_text())
        del meta["model_dirs"]
        (directory / "metadata.json").write_text(json.dumps(meta))

        assert ExperimentResult.load(directory).model_names == [REFERENCE_MODEL, "ElasticNet"]

    def test_load_backfills_a_missing_split_mode_from_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        runs = [make_run_result(model_name="A", split_mode="LTO"), make_run_result(model_name="B", split_mode="")]
        ExperimentResult(runs).save(directory)

        loaded = ExperimentResult.load(directory)

        assert {r.split_mode for m in loaded.models for r in m.runs} == {"LTO"}


class TestFailures:
    FAILURES = (
        RunFailure("B", 1, None, None, "ValueError", "boom"),
        RunFailure("B", 0, ("SVRC", "gene_expression"), 2, "KeyError", "'gene'"),
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
        assert json.loads((directory / "failures.json").read_text())[1]["randomization"] == ["SVRC", "gene_expression"]
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
        run.sample_ids = intern_ids(run.sample_ids)
        run.drug_ids = intern_ids(run.drug_ids)
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

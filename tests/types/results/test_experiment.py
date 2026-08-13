"""Tests for the experiment-level result: guards, capabilities and normalization."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from upath import UPath

from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.types.results.experiment import ExperimentResult, _run_array_bytes
from drevalpy.types.results.run import intern_ids
from drevalpy.types.results.trial import TrialResult
from drevalpy.visualization.requirements import PlotRequirement
from tests.synthetic import NORMALIZED_METRIC, REFERENCE_MODEL, make_experiment_result, make_run_result


class TestConstruction:
    def test_rejects_an_empty_run_list(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ExperimentResult([])

    def test_rejects_mixed_dataset_names(self) -> None:
        runs = [
            make_run_result(dataset_name="GDSC1"),
            make_run_result(dataset_name="CTRPv2"),
        ]

        with pytest.raises(ValueError, match="same dataset_name"):
            ExperimentResult(runs)

    def test_rejects_mixed_split_modes(self) -> None:
        runs = [
            make_run_result(model_name="A", split_mode="LPO"),
            make_run_result(model_name="B", split_mode="LCO"),
        ]

        with pytest.raises(ValueError, match="same split_mode"):
            ExperimentResult(runs)

    def test_ignores_blank_split_modes_when_checking_consistency(self) -> None:
        runs = [
            make_run_result(model_name="A", split_mode="LPO"),
            make_run_result(model_name="B", split_mode=""),
        ]

        assert ExperimentResult(runs).split_mode == "LPO"

    def test_split_mode_is_blank_when_no_run_declares_one(self) -> None:
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

    def test_propagates_dataset_name_to_each_model(self) -> None:
        result = make_experiment_result(n_models=2, n_folds=1)

        assert {m.dataset_name for m in result.models} == {"SyntheticDataset"}

    def test_starts_unnormalized(self) -> None:
        assert make_experiment_result(n_models=1, n_folds=1).normalized_by is None


class TestCapabilities:
    def test_counts_distinct_models(self) -> None:
        assert make_experiment_result(n_models=3, n_folds=2).n_models == 3

    def test_max_folds_takes_the_largest_model(self) -> None:
        runs = [
            make_run_result(model_name="A", fold_index=0),
            make_run_result(model_name="B", fold_index=0),
            make_run_result(model_name="B", fold_index=1),
        ]

        assert ExperimentResult(runs).max_folds == 2

    def test_randomization_is_absent_by_default(self) -> None:
        assert make_experiment_result(n_models=2, n_folds=1).has_randomization is False

    def test_randomization_is_detected(self) -> None:
        result = make_experiment_result(n_models=2, n_folds=1, with_randomization=True)

        assert result.has_randomization is True

    def test_robustness_is_absent_by_default(self) -> None:
        assert make_experiment_result(n_models=2, n_folds=1).has_robustness is False

    def test_robustness_is_detected_from_fold_metadata(self) -> None:
        result = make_experiment_result(n_models=2, n_folds=1, with_robustness=True)

        assert result.has_robustness is True

    def test_summary_table_holds_the_mean_of_each_metric(self) -> None:
        runs = [
            make_run_result(model_name="A", fold_index=0, metrics={"MSE": 0.2}),
            make_run_result(model_name="A", fold_index=1, metrics={"MSE": 0.4}),
        ]

        table = ExperimentResult(runs).summary_table

        assert table == {"A": {"MSE": pytest.approx(0.3)}}


class TestSatisfies:
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

        requirements = frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})
        assert result.satisfies(requirements) is True

    def test_randomization_requirement_needs_randomization_data(self) -> None:
        requirements = frozenset({PlotRequirement.RANDOMIZATION})

        assert make_experiment_result(n_models=2, n_folds=2).satisfies(requirements) is False
        assert make_experiment_result(n_models=2, n_folds=2, with_randomization=True).satisfies(requirements) is True

    def test_robustness_requirement_needs_robustness_data(self) -> None:
        requirements = frozenset({PlotRequirement.ROBUSTNESS})

        assert make_experiment_result(n_models=2, n_folds=2).satisfies(requirements) is False
        assert make_experiment_result(n_models=2, n_folds=2, with_robustness=True).satisfies(requirements) is True


class TestNormalize:
    def test_drops_the_reference_model(self) -> None:
        normalized = make_experiment_result(n_models=3, n_folds=2).normalize()

        assert REFERENCE_MODEL not in normalized.model_names
        assert normalized.n_models == 2

    def test_records_the_reference_model(self) -> None:
        assert make_experiment_result(n_models=2, n_folds=1).normalize().normalized_by == REFERENCE_MODEL

    def test_subtracts_reference_predictions_pairwise(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)
        reference, other = experiment.models

        normalized_run = experiment.normalize().models[0].runs[0]

        expected = other.runs[0].predictions - reference.runs[0].predictions
        np.testing.assert_allclose(normalized_run.predictions, expected)

    def test_recomputes_only_the_standard_metrics(self) -> None:
        normalized = make_experiment_result(n_models=2, n_folds=1).normalize()

        metrics = normalized.models[0].runs[0].metrics
        assert set(metrics) == set(AVAILABLE_METRICS)
        assert NORMALIZED_METRIC not in metrics

    def test_treats_pairs_missing_from_the_reference_as_zero(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=4, n_cell_lines=2, n_drugs=2)
        other = make_run_result(model_name="ElasticNet", n_pairs=4, n_cell_lines=2, n_drugs=2)
        other.cell_line_ids = np.array(["CL_9", "CL_9", "CL_9", "CL_9"])

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        np.testing.assert_allclose(normalized_run.predictions, other.predictions)

    def test_yields_no_metrics_when_every_pair_is_nan(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=4)
        other = make_run_result(model_name="ElasticNet", n_pairs=4)
        other.predictions = np.full(4, np.nan)

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        assert normalized_run.metrics == {}

    def test_carries_trials_over(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=4)
        other = make_run_result(model_name="ElasticNet", n_pairs=4)
        other.trials = [
            TrialResult(
                hyperparameters={"alpha": 0.1},
                metrics={"MSE": 0.3},
                optimization_metric="MSE",
                predictions=np.zeros(4),
            )
        ]

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        assert normalized_run.trials is not None
        assert normalized_run.trials[0].hyperparameters == {"alpha": 0.1}

    def test_preserves_run_identity(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=2, split_mode="LCO")

        normalized_run = experiment.normalize().models[0].runs[1]

        assert normalized_run.fold_id == "fold_1"
        assert normalized_run.fold_index == 1
        assert normalized_run.split_mode == "LCO"

    def test_rejects_a_second_normalization(self) -> None:
        normalized = make_experiment_result(n_models=2, n_folds=1).normalize()

        with pytest.raises(ValueError, match="Already normalized"):
            normalized.normalize()

    def test_rejects_an_unknown_reference_model(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)

        with pytest.raises(ValueError, match="not found"):
            experiment.normalize(reference_model="NotAModel")

    def test_rejects_a_fold_without_a_reference_run(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, fold_index=0)
        other = make_run_result(model_name="ElasticNet", fold_index=1)

        with pytest.raises(ValueError, match="No reference run"):
            ExperimentResult([reference, other]).normalize()

    def test_accepts_an_explicit_reference_model(self) -> None:
        experiment = make_experiment_result(n_models=3, n_folds=1)

        normalized = experiment.normalize(reference_model="ElasticNet")

        assert normalized.normalized_by == "ElasticNet"
        assert "ElasticNet" not in normalized.model_names

    def test_normalizes_interned_id_arrays(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=8)
        other = make_run_result(model_name="ElasticNet", n_pairs=8)
        for run in (reference, other):
            run.cell_line_ids = intern_ids(run.cell_line_ids)
            run.drug_ids = intern_ids(run.drug_ids)

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        np.testing.assert_allclose(normalized_run.predictions, other.predictions - reference.predictions)

    def test_normalizes_only_the_pairs_the_reference_covers(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=4, n_cell_lines=2, n_drugs=2)
        other = make_run_result(model_name="ElasticNet", n_pairs=4, n_cell_lines=2, n_drugs=2)
        other.cell_line_ids = np.array(["CL_0", "CL_9", "CL_0", "CL_9"])
        other.drug_ids = np.array(["D_0", "D_0", "D_1", "D_1"])

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        ref_by_pair = dict(
            zip(
                zip(reference.cell_line_ids, reference.drug_ids, strict=True),
                reference.predictions,
                strict=True,
            )
        )
        expected = other.predictions - np.array(
            [ref_by_pair.get((cl, dr), 0.0) for cl, dr in zip(other.cell_line_ids, other.drug_ids, strict=True)]
        )
        np.testing.assert_allclose(normalized_run.predictions, expected)

    def test_a_duplicated_reference_pair_resolves_to_its_last_prediction(self) -> None:
        reference = make_run_result(model_name=REFERENCE_MODEL, n_pairs=2)
        reference.cell_line_ids = np.array(["CL_0", "CL_0"])
        reference.drug_ids = np.array(["D_0", "D_0"])
        reference.predictions = np.array([1.0, 5.0])
        other = make_run_result(model_name="ElasticNet", n_pairs=1)
        other.cell_line_ids = np.array(["CL_0"])
        other.drug_ids = np.array(["D_0"])
        other.predictions = np.array([7.0])
        other.ground_truth = np.array([7.0])

        normalized_run = ExperimentResult([reference, other]).normalize().models[0].runs[0]

        np.testing.assert_allclose(normalized_run.predictions, np.array([2.0]))

    def test_normalizes_ground_truth_by_the_same_offset(self) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)
        reference, other = experiment.models

        normalized_run = experiment.normalize().models[0].runs[0]

        expected = other.runs[0].ground_truth - reference.runs[0].predictions
        np.testing.assert_allclose(normalized_run.ground_truth, expected)


class TestPersistence:
    def test_save_writes_one_directory_per_model(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"

        make_experiment_result(n_models=2, n_folds=1).save(directory)

        assert (directory / REFERENCE_MODEL / "metadata.json").is_file()
        assert (directory / "ElasticNet" / "metadata.json").is_file()

    def test_save_records_experiment_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"

        make_experiment_result(n_models=2, n_folds=1, split_mode="LDO").save(directory)
        meta = json.loads((directory / "metadata.json").read_text())

        assert meta["dataset_name"] == "SyntheticDataset"
        assert meta["split_mode"] == "LDO"
        assert meta["normalized_by"] is None
        assert meta["models"] == [REFERENCE_MODEL, "ElasticNet"]

    def test_round_trip_preserves_structure(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        experiment = make_experiment_result(n_models=3, n_folds=2)

        experiment.save(directory)
        loaded = ExperimentResult.load(directory)

        assert loaded.model_names == experiment.model_names
        assert loaded.max_folds == 2
        assert loaded.split_mode == experiment.split_mode
        assert loaded.normalized_by is None

    def test_round_trip_preserves_the_reference_model(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=2, n_folds=1).normalize().save(directory)

        assert ExperimentResult.load(directory).normalized_by == REFERENCE_MODEL

    def test_load_backfills_a_missing_split_mode_from_metadata(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        runs = [
            make_run_result(model_name="A", split_mode="LTO"),
            make_run_result(model_name="B", split_mode=""),
        ]
        ExperimentResult(runs).save(directory)

        loaded = ExperimentResult.load(directory)

        assert {r.split_mode for m in loaded.models for r in m.runs} == {"LTO"}

    def test_accepts_a_plain_string_directory(self, tmp_path) -> None:
        directory = str(UPath(tmp_path) / "experiment")

        make_experiment_result(n_models=1, n_folds=1).save(directory)

        assert ExperimentResult.load(directory).n_models == 1


class TestTrialSkipping:
    """The report path loads without trials; every other caller keeps the default."""

    @staticmethod
    def _saved_with_trials(directory: UPath) -> None:
        experiment = make_experiment_result(n_models=2, n_folds=1)
        for model in experiment.models:
            for run in model.runs:
                run.trials = [
                    TrialResult(
                        hyperparameters={"alpha": 0.1},
                        metrics={"MSE": 0.3},
                        optimization_metric="MSE",
                        predictions=np.zeros(4),
                    )
                ]
        experiment.save(directory)

    def test_trials_are_loaded_by_default(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        self._saved_with_trials(directory)

        loaded = ExperimentResult.load(directory)

        assert all(run.trials for model in loaded.models for run in model.runs)

    def test_trials_can_be_skipped(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        self._saved_with_trials(directory)

        loaded = ExperimentResult.load(directory, with_trials=False)

        assert all(run.trials is None for model in loaded.models for run in model.runs)

    def test_skipping_trials_preserves_the_structure(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "experiment"
        self._saved_with_trials(directory)

        loaded = ExperimentResult.load(directory, with_trials=False)

        assert loaded.model_names == [REFERENCE_MODEL, "ElasticNet"]
        assert loaded.max_folds == 1


class TestLoadLogging:
    """The load summary is the line that establishes the scale of a run."""

    def test_reports_models_runs_rows_and_bytes(self, tmp_path, caplog) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=2, n_folds=2, n_pairs=20).save(directory)

        with caplog.at_level(logging.INFO, logger="drevalpy.types.results.experiment"):
            ExperimentResult.load(directory)

        message = caplog.records[-1].getMessage()
        assert "2 models" in message
        assert "4 runs" in message
        assert "80 prediction rows" in message
        assert "GB of arrays" in message

    def test_records_whether_trials_were_skipped(self, tmp_path, caplog) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=1, n_folds=1).save(directory)

        with caplog.at_level(logging.INFO, logger="drevalpy.types.results.experiment"):
            ExperimentResult.load(directory, with_trials=False)

        assert "trials skipped" in caplog.records[-1].getMessage()

    def test_records_when_trials_were_loaded(self, tmp_path, caplog) -> None:
        directory = UPath(tmp_path) / "experiment"
        make_experiment_result(n_models=1, n_folds=1).save(directory)

        with caplog.at_level(logging.INFO, logger="drevalpy.types.results.experiment"):
            ExperimentResult.load(directory)

        assert "trials loaded" in caplog.records[-1].getMessage()

    def test_counts_trial_predictions_in_the_byte_total(self, tmp_path) -> None:
        run = make_run_result(n_pairs=8)
        without = _run_array_bytes(run)
        run.trials = [
            TrialResult(
                hyperparameters={},
                metrics={},
                optimization_metric="MSE",
                predictions=np.zeros(1000),
            )
        ]

        assert _run_array_bytes(run) > without

    def test_interned_ids_count_their_shared_strings(self) -> None:
        run = make_run_result(n_pairs=8)
        run.cell_line_ids = intern_ids(run.cell_line_ids)
        run.drug_ids = intern_ids(run.drug_ids)

        assert _run_array_bytes(run) > run.predictions.nbytes + run.ground_truth.nbytes


class TestRepr:
    def test_reports_experiment_level_metadata(self) -> None:
        text = repr(make_experiment_result(n_models=2, n_folds=2, split_mode="LCO"))

        assert text.startswith("ExperimentResult")
        assert "Dataset: SyntheticDataset" in text
        assert "Split mode: LCO" in text
        assert "Normalized by: None" in text
        assert "Models: 2" in text

    def test_reports_metrics_per_model(self) -> None:
        runs = [make_run_result(model_name="A", metrics={"MSE": 0.5})]

        assert "        A (1 folds): MSE=0.5000" in repr(ExperimentResult(runs)).splitlines()

    def test_reports_models_without_metrics(self) -> None:
        runs = [make_run_result(model_name="A", metrics={})]

        assert "        A (1 folds): no metrics" in repr(ExperimentResult(runs)).splitlines()

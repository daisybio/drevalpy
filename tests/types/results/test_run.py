"""Tests for the per-fold run result dataclass and its npz round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
from upath import UPath

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


class TestDefaults:
    def test_optional_metadata_defaults_to_empty(self) -> None:
        run = RunResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            fold_index=0,
            predictions=np.zeros(3),
            ground_truth=np.zeros(3),
            cell_line_ids=np.array(["CL_0", "CL_1", "CL_2"]),
            drug_ids=np.array(["D_0", "D_1", "D_2"]),
        )

        assert run.split_mode == ""
        assert run.fold_id == ""
        assert run.best_hyperparameters == {}
        assert run.metrics == {}
        assert run.fold_metadata == {}
        assert run.trials is None
        assert run.randomization is None

    def test_builder_derives_fold_id_from_fold_index(self) -> None:
        assert make_run_result(fold_index=2).fold_id == "fold_2"

    def test_builder_metrics_cover_every_reported_metric(self) -> None:
        assert make_run_result().metrics.keys() == make_metrics().keys()


class TestRepr:
    def test_reports_model_and_dataset(self) -> None:
        text = repr(make_run_result(model_name="RandomForest"))

        assert text.startswith("RunResult")
        assert "Model: RandomForest" in text
        assert "Dataset: SyntheticDataset" in text

    def test_reports_absent_randomization_explicitly(self) -> None:
        assert "Randomization: None" in repr(make_run_result())

    def test_reports_randomization_mode_and_view(self) -> None:
        run = make_run_result(randomization=("SVRC", "gene_expression"))

        assert "Randomization: SVRC (gene_expression)" in repr(run)

    def test_omits_fold_index_from_the_metadata_block(self) -> None:
        run = make_run_result(fold_index=1, fold_metadata={"fold_index": 1, "robustness_trial": 3})

        lines = repr(run).splitlines()

        assert "        robustness_trial: 3" in lines
        assert "        fold_index: 1" not in lines

    def test_counts_only_non_nan_ground_truth(self) -> None:
        run = make_run_result(n_pairs=5)
        run.ground_truth = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

        assert "Ground truth: 3 non-NaN values" in repr(run)

    def test_reports_pair_count(self) -> None:
        assert "Predictions: 7 pairs" in repr(make_run_result(n_pairs=7))

    def test_lists_hyperparameters_and_metrics(self) -> None:
        run = make_run_result(best_hyperparameters={"alpha": 0.25}, metrics={"MSE": 0.5})

        lines = repr(run).splitlines()

        assert "    Hyperparameters:" in lines
        assert "        alpha: 0.25" in lines
        assert "        MSE: 0.5000" in lines

    def test_omits_empty_hyperparameter_and_metric_blocks(self) -> None:
        run = RunResult(
            model_name="ElasticNet",
            dataset_name="SyntheticDataset",
            fold_index=0,
            predictions=np.zeros(3),
            ground_truth=np.zeros(3),
            cell_line_ids=np.array(["CL_0", "CL_1", "CL_2"]),
            drug_ids=np.array(["D_0", "D_1", "D_2"]),
        )

        text = repr(run)

        assert "Hyperparameters:" not in text
        assert "Metrics:" not in text

    def test_reports_trial_count_when_present(self) -> None:
        run = make_run_result()
        run.trials = [_trial(0), _trial(1)]

        assert "HPO Trials: 2" in repr(run)

    def test_omits_trial_line_without_trials(self) -> None:
        assert "HPO Trials" not in repr(make_run_result())


class TestPersistence:
    def test_round_trip_preserves_arrays(self, tmp_path) -> None:
        run = make_run_result(n_pairs=9)
        path = UPath(tmp_path) / "fold_0.npz"

        run.save(path)
        loaded = RunResult.load(path)

        np.testing.assert_allclose(loaded.predictions, run.predictions)
        np.testing.assert_allclose(loaded.ground_truth, run.ground_truth)
        np.testing.assert_array_equal(loaded.cell_line_ids, run.cell_line_ids)
        np.testing.assert_array_equal(loaded.drug_ids, run.drug_ids)

    def test_round_trip_preserves_scalar_metadata(self, tmp_path) -> None:
        run = make_run_result(model_name="RandomForest", fold_index=2, split_mode="LCO")
        path = UPath(tmp_path) / "fold_2.npz"

        run.save(path)
        loaded = RunResult.load(path)

        assert loaded.model_name == "RandomForest"
        assert loaded.dataset_name == run.dataset_name
        assert loaded.split_mode == "LCO"
        assert loaded.fold_index == 2
        assert loaded.fold_id == "fold_2"

    def test_round_trip_preserves_metric_and_hyperparameter_dicts(self, tmp_path) -> None:
        run = make_run_result(best_hyperparameters={"alpha": 0.25}, fold_metadata={"note": "kept"})
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)
        loaded = RunResult.load(path)

        assert loaded.metrics == pytest.approx(run.metrics)
        assert loaded.best_hyperparameters == {"alpha": 0.25}
        assert loaded.fold_metadata == {"note": "kept"}

    def test_round_trip_restores_randomization_as_a_tuple(self, tmp_path) -> None:
        run = make_run_result(randomization=("SVRC", "gene_expression"))
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)
        loaded = RunResult.load(path)

        assert loaded.randomization == ("SVRC", "gene_expression")

    def test_round_trip_keeps_randomization_none(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result().save(path)

        assert RunResult.load(path).randomization is None

    def test_round_trip_restores_every_trial(self, tmp_path) -> None:
        run = make_run_result()
        run.trials = [_trial(0), _trial(1)]
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)
        loaded = RunResult.load(path)

        assert loaded.trials is not None
        assert [t.hyperparameters for t in loaded.trials] == [t.hyperparameters for t in run.trials]
        assert [t.optimization_metric for t in loaded.trials] == ["MSE", "MSE"]
        np.testing.assert_allclose(loaded.trials[1].predictions, np.full(4, 1.0))

    def test_round_trip_keeps_trials_none(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result().save(path)

        assert RunResult.load(path).trials is None

    def test_empty_trial_list_is_not_serialized(self, tmp_path) -> None:
        run = make_run_result()
        run.trials = []
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)

        with np.load(path, allow_pickle=False) as data:
            assert json.loads(str(data["_metadata"]))["trials"] is None

    def test_accepts_a_plain_string_path(self, tmp_path) -> None:
        path = str(UPath(tmp_path) / "fold.npz")
        make_run_result(fold_index=1).save(path)

        assert RunResult.load(path).fold_index == 1

    def test_entity_ids_are_stored_as_strings(self, tmp_path) -> None:
        run = make_run_result(n_pairs=3)
        run.cell_line_ids = np.array([1, 2, 3])
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)

        np.testing.assert_array_equal(RunResult.load(path).cell_line_ids, np.array(["1", "2", "3"]))

    def test_metadata_blob_is_json(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result(split_mode="LDO").save(path)

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["_metadata"]))

        assert meta["split_mode"] == "LDO"
        assert meta["fold_id"] == "fold_0"


class TestTrialSkipping:
    """``with_trials=False`` lets the report path avoid the largest arrays in the file."""

    def test_trials_are_loaded_by_default(self, tmp_path) -> None:
        run = make_run_result()
        run.trials = [_trial(0), _trial(1)]
        path = UPath(tmp_path) / "fold.npz"
        run.save(path)

        assert RunResult.load(path).trials is not None

    def test_trials_can_be_skipped(self, tmp_path) -> None:
        run = make_run_result()
        run.trials = [_trial(0), _trial(1)]
        path = UPath(tmp_path) / "fold.npz"
        run.save(path)

        assert RunResult.load(path, with_trials=False).trials is None

    def test_skipping_trials_leaves_the_fold_arrays_intact(self, tmp_path) -> None:
        run = make_run_result(n_pairs=6)
        run.trials = [_trial(0)]
        path = UPath(tmp_path) / "fold.npz"
        run.save(path)

        loaded = RunResult.load(path, with_trials=False)

        np.testing.assert_allclose(loaded.predictions, run.predictions)
        np.testing.assert_array_equal(loaded.drug_ids, run.drug_ids)


class TestInternIds:
    """Entity ids are deduplicated at load; ``<U40`` costs 160 B per element otherwise."""

    def test_values_are_preserved(self) -> None:
        ids = np.array(["CL_0", "CL_1", "CL_0"], dtype="<U40")

        np.testing.assert_array_equal(intern_ids(ids), ids)

    def test_the_result_is_an_object_array(self) -> None:
        assert intern_ids(np.array(["CL_0", "CL_1"])).dtype == object

    def test_indexing_still_yields_a_str(self) -> None:
        assert intern_ids(np.array(["CL_0"]))[0] == "CL_0"
        assert isinstance(intern_ids(np.array(["CL_0"]))[0], str)

    def test_repeated_ids_share_one_string_object(self) -> None:
        interned = intern_ids(np.array(["CL_0"] * 5))

        assert len({id(value) for value in interned.tolist()}) == 1

    def test_an_empty_array_stays_empty(self) -> None:
        assert intern_ids(np.array([], dtype="<U40")).shape == (0,)

    def test_the_pointer_table_is_smaller_than_fixed_width_storage(self) -> None:
        ids = np.array([f"BRD-K{i:035d}" for i in range(4)] * 250, dtype="<U40")

        assert intern_ids(ids).nbytes < ids.nbytes

    def test_load_interns_both_id_arrays(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result(n_pairs=6).save(path)

        loaded = RunResult.load(path)

        assert loaded.cell_line_ids.dtype == object
        assert loaded.drug_ids.dtype == object

    def test_the_save_load_round_trip_stays_lossless_through_interning(self, tmp_path) -> None:
        run = make_run_result(n_pairs=8, n_cell_lines=3, n_drugs=2)
        first = UPath(tmp_path) / "a.npz"
        second = UPath(tmp_path) / "b.npz"

        run.save(first)
        once = RunResult.load(first)
        once.save(second)
        twice = RunResult.load(second)

        np.testing.assert_array_equal(once.cell_line_ids, run.cell_line_ids)
        np.testing.assert_array_equal(twice.drug_ids, run.drug_ids)

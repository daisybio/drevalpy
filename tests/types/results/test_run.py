"""Per-fold run results: defaults, repr and persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest
from upath import UPath

from drevalpy.types.results.run import RunResult, _intern_ids
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


class TestDefaults:
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


class TestRepr:
    def test_reports_the_run_summary(self) -> None:
        run = make_run_result(
            model_name="RandomForest",
            n_pairs=5,
            fold_index=1,
            fold_metadata={"fold_index": 1, "robustness_repeat": 3},
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
        assert "        robustness_repeat: 3" in lines
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

    def test_the_fold_fields_are_stored_once_not_again_in_fold_metadata(self, tmp_path) -> None:
        run = make_run_result(
            fold_index=3,
            split_mode="LSO",
            fold_metadata={"fold_index": 3, "fold_id": "fold_3", "split_mode": "LSO", "dataset": "d", "note": "kept"},
        )
        path = UPath(tmp_path) / "fold.npz"

        run.save(path)

        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["_metadata"]))
        assert meta["fold_metadata"] == {"note": "kept"}
        assert (meta["fold_index"], meta["fold_id"], meta["split_mode"], meta["dataset_name"]) == (
            3,
            "fold_3",
            "LSO",
            "SyntheticDataset",
        )
        assert set(meta) == {
            "model_name",
            "dataset_name",
            "split_mode",
            "fold_index",
            "fold_id",
            "best_hyperparameters",
            "metrics",
            "fold_metadata",
            "randomization",
            "trials",
            "raw_metrics",
            "raw_per_drug_pearson",
        }

    def test_the_file_is_closed_after_load_and_after_save(self, tmp_path) -> None:
        path = UPath(tmp_path) / "fold.npz"
        make_run_result().save(path)
        RunResult.load(path)

        path.unlink()

        assert not path.exists()


class TestInternIds:
    def test_values_are_preserved_as_shared_str_objects(self) -> None:
        ids = np.array(["CL_0", "CL_1", "CL_0"] + ["CL_0"] * 5, dtype="<U40")

        interned = _intern_ids(ids)

        np.testing.assert_array_equal(interned, ids)
        assert interned.dtype == object
        assert isinstance(interned[0], str)
        assert len({id(value) for value in interned.tolist() if value == "CL_0"}) == 1

    def test_empty_and_pointer_table_size(self) -> None:
        assert _intern_ids(np.array([], dtype="<U40")).shape == (0,)
        ids = np.array([f"BRD-K{i:035d}" for i in range(4)] * 250, dtype="<U40")
        assert _intern_ids(ids).nbytes < ids.nbytes

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

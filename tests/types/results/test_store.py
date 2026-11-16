"""Tests for the experiment layout and the ``RunStore`` that ``run`` saves each run into."""

from __future__ import annotations

import json

import numpy as np
import pytest
from upath import UPath

from drevalpy.errors import OutputDirectoryConflictError
from drevalpy.types.results._store import RunStore, run_filename
from drevalpy.types.results.experiment import ExperimentResult
from tests.synthetic import make_run_failure, make_run_result

SETTINGS = {"models": [{"name": "ElasticNet"}, {"name": "pca[gene_expression]:fingerprints:ridge"}], "seed": 42}


def _open(directory: UPath, settings: dict | None = None, *, overwrite: bool = False) -> RunStore:
    return RunStore.open(directory, SETTINGS if settings is None else settings, overwrite=overwrite)


class TestRunFilename:
    def test_the_name_is_a_readable_label_and_a_digest_of_the_checkpoint_key(self) -> None:
        plain = make_run_result(fold_index=3)
        stressed = make_run_result(
            fold_index=3, fold_metadata={"robustness_repeat": 1}, randomization=("SVRC", "gene_expression")
        )

        assert run_filename(plain.identity).startswith("fold-3_")
        assert run_filename(stressed.identity).startswith("fold-3_SVRC_gene_expression_repeat-1_")
        assert run_filename(plain.identity).endswith(".npz")
        assert run_filename(plain.identity) != run_filename(stressed.identity)

    def test_the_provenance_does_not_change_the_name(self) -> None:
        bare = make_run_result()
        fitted = make_run_result(fold_metadata={"execution": {"seed": 7}})

        assert bare.run_key != fitted.run_key
        assert run_filename(bare.identity) == run_filename(fitted.identity)


class TestOpen:
    def test_a_new_directory_records_the_settings_and_the_model_order(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "new" / "results"

        _open(directory)

        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert meta == {
            "normalized_by": None,
            "models": ["ElasticNet", "pca[gene_expression]:fingerprints:ridge"],
            "settings": SETTINGS,
        }

    def test_an_empty_existing_directory_is_started(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        directory.mkdir()

        _open(directory)

        assert (directory / "metadata.json").is_file()

    def test_the_same_settings_resume_and_other_settings_raise(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        _open(directory).save(make_run_result())

        assert _open(directory).load(make_run_result().identity) is not None
        with pytest.raises(OutputDirectoryConflictError, match="other settings.*--overwrite.*overwrite=True"):
            _open(directory, {**SETTINGS, "seed": 1})

    @pytest.mark.parametrize(
        "metadata",
        [None, "{not json", "[1, 2]", json.dumps({"models": "ElasticNet", "normalized_by": None})],
        ids=["none", "unreadable", "not-an-object", "not-a-model-list"],
    )
    @pytest.mark.parametrize("overwrite", [False, True])
    def test_files_of_no_experiment_are_refused_and_never_deleted(
        self, tmp_path, metadata: str | None, overwrite: bool
    ) -> None:
        directory = UPath(tmp_path) / "results"
        directory.mkdir()
        (directory / "notes.txt").write_text("mine", encoding="utf-8")
        if metadata is not None:
            (directory / "metadata.json").write_text(metadata, encoding="utf-8")

        with pytest.raises(OutputDirectoryConflictError, match="no drevalpy run wrote.*empty or new directory"):
            _open(directory, overwrite=overwrite)
        assert (directory / "notes.txt").read_text(encoding="utf-8") == "mine"

    def test_an_experiment_without_runs_is_started_again_with_the_new_settings(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        _open(directory).finish([make_run_failure("ElasticNet", 0)])

        _open(directory, {**SETTINGS, "seed": 1})

        assert json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["settings"]["seed"] == 1
        assert not (directory / "failures.json").exists()

    def test_runs_saved_without_settings_are_refused(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        ExperimentResult([make_run_result()]).save(directory)

        with pytest.raises(OutputDirectoryConflictError, match="other settings"):
            _open(directory)

    def test_overwrite_deletes_only_the_experiment_and_starts_again(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        store = _open(directory)
        store.save(make_run_result())
        store.finish([make_run_failure("ElasticNet", 1)])
        (directory / "notes.txt").write_text("mine", encoding="utf-8")
        (directory / "plots").mkdir()

        store = _open(directory, {**SETTINGS, "seed": 1}, overwrite=True)

        assert store.load(make_run_result().identity) is None
        assert sorted(entry.name for entry in directory.iterdir()) == ["metadata.json", "notes.txt", "plots"]
        assert json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["settings"]["seed"] == 1


class TestRuns:
    def test_save_load_and_finish_round_trip(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        store = _open(directory)
        runs = [make_run_result(fold_index=i) for i in range(2)]
        runs.append(make_run_result(model_name="pca[gene_expression]:fingerprints:ridge"))
        failure = make_run_failure("ElasticNet", 2)

        for run in runs:
            assert store.load(run.identity) is None
            store.save(run)
        store.finish([failure])

        loaded = store.load(runs[1].identity)
        assert loaded is not None
        np.testing.assert_array_equal(loaded.predictions, runs[1].predictions)
        assert (directory / "ElasticNet" / run_filename(runs[0].identity)).is_file()
        assert not list(directory.rglob("*.partial"))
        experiment = ExperimentResult.load(directory)
        assert [[r.run_key for r in m.runs] for m in experiment.models] == [
            [r.run_key for r in runs[:2]],
            [runs[2].run_key],
        ]
        assert experiment.failures == (failure,)

    def test_finish_removes_a_stale_failures_file_and_keeps_the_settings(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        run = make_run_result()
        store = _open(directory)
        store.save(run)
        store.finish([make_run_failure("ElasticNet", 1)])

        resumed = _open(directory)
        resumed.finish([])

        assert not (directory / "failures.json").exists()
        assert json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["settings"] == SETTINGS
        assert ExperimentResult.load(directory).failures == ()

    def test_a_killed_run_leaves_a_loadable_directory_of_the_finished_runs(self, tmp_path) -> None:
        directory = UPath(tmp_path) / "results"
        _open(directory).save(make_run_result())

        assert ExperimentResult.load(directory).model_names == ["ElasticNet"]

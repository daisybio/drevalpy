"""``--group-by`` with a sample or drug annotation column in ``drevalpy run`` and ``drevalpy aggregate``."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from upath import UPath

from drevalpy.cli._summary import format_grouped_summary
from drevalpy.cli.main import app
from drevalpy.data import load
from drevalpy.types import Dataset
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import Recorder, make_runner, patch_worker, plain
from tests.synthetic import make_run_result

runner = make_runner()

HEADING = (
    "Metrics computed within each {unit}, averaged over {plural} and folds (number of {plural} scored in brackets):"
)


@pytest.fixture(scope="module")
def dataset(dataset_file: str) -> Dataset:
    return Dataset.load(dataset_file)


def _run_file(dataset: Dataset, path: UPath, model_name: str = "ElasticNet", fold_index: int = 0) -> UPath:
    """Save a run over the sample-drug pairs of the dataset."""
    run = make_run_result(model_name=model_name, fold_index=fold_index, n_pairs=40)
    run.sample_ids = np.array([dataset.sample_ids[i % len(dataset.sample_ids)] for i in range(40)])
    run.drug_ids = np.array([dataset.drug_ids[i // len(dataset.sample_ids)] for i in range(40)])
    run.save(path)
    return path


@pytest.fixture()
def run_files(dataset: Dataset, tmp_path: UPath) -> list[UPath]:
    return [
        _run_file(dataset, tmp_path / f"{name}_{fold}.npz", name, fold)
        for name in ("ElasticNet", "RandomForest")
        for fold in range(2)
    ]


def _aggregate(run_files: list[UPath], out_dir: UPath, *extra: str) -> Any:
    return runner.invoke(app, ["aggregate", *[str(path) for path in run_files], "-o", str(out_dir), *extra])


class TestAggregate:
    @pytest.mark.parametrize(
        ("group_by", "unit"),
        [("tissue", "tissue group"), ("sample:tissue", "tissue group"), ("drug:drug_name", "drug_name group")],
    )
    def test_an_annotation_grouping_prints_the_summary(
        self, run_files: list[UPath], dataset_file: str, tmp_path: UPath, group_by: str, unit: str
    ) -> None:
        result = _aggregate(run_files, tmp_path / "out", "--group-by", group_by, "--dataset", dataset_file)

        assert result.exit_code == 0, result.output
        text = plain(result.output)
        assert HEADING.format(unit=unit, plural="groups") in text
        assert "ElasticNet" in text.split("Metrics computed within")[1]

    def test_the_scores_are_those_of_the_annotation_groups(
        self, run_files: list[UPath], dataset_file: str, dataset: Dataset, tmp_path: UPath
    ) -> None:
        result = _aggregate(run_files, tmp_path / "out", "--group-by", "tissue", "-d", dataset_file)

        experiment = ExperimentResult.load(tmp_path / "out")
        counts = experiment.summary_group_counts("tissue", dataset)
        expected = format_grouped_summary(experiment, "tissue", dataset)
        assert expected in plain(result.output)
        assert counts["MSE"].tolist() == [len(set(dataset.sample_meta["tissue"]))] * 2

    def test_an_annotation_without_a_dataset_is_a_usage_error_and_writes_nothing(
        self, run_files: list[UPath], tmp_path: UPath
    ) -> None:
        result = _aggregate(run_files, tmp_path / "out", "--group-by", "tissue")

        assert result.exit_code == 2
        assert "--dataset" in plain(result.output)
        assert not (tmp_path / "out").exists()

    def test_an_unknown_column_is_a_usage_error_listing_the_annotations(
        self, run_files: list[UPath], dataset_file: str, tmp_path: UPath
    ) -> None:
        result = _aggregate(run_files, tmp_path / "out", "--group-by", "nope", "--dataset", dataset_file)

        assert result.exit_code == 2
        text = " ".join(plain(result.output).split())
        assert "Unknown grouping 'nope'" in text
        assert "'tissue'" in text
        assert "'drug_name'" in text
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize("group_by", ["drug", "SAMPLE", "study"])
    def test_a_builtin_grouping_never_reads_the_dataset(
        self, run_files: list[UPath], tmp_path: UPath, group_by: str
    ) -> None:
        result = _aggregate(run_files, tmp_path / "out", "--group-by", group_by, "-d", str(tmp_path / "missing.h5mu"))

        assert result.exit_code == 0, result.output

    def test_the_dataset_is_ignored_without_a_grouping(self, run_files: list[UPath], tmp_path: UPath) -> None:
        result = _aggregate(run_files, tmp_path / "out", "-d", str(tmp_path / "missing.h5mu"))

        assert result.exit_code == 0, result.output
        assert "Metrics computed within" not in plain(result.output)


class TestRun:
    def test_an_annotation_grouping_scores_the_run_without_loading_the_dataset_twice(
        self, dataset_file: str, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = importlib.import_module("drevalpy._run")
        loaded: list[str] = []

        def counting_load(name: str) -> Dataset:
            loaded.append(name)
            return load(name)

        monkeypatch.setattr(module, "load", counting_load)

        result = runner.invoke(
            app,
            [
                "run",
                "NaiveMeanEffectsPredictor",
                "-d",
                dataset_file,
                "--no-hpo",
                "-o",
                str(tmp_path / "results"),
                "--group-by",
                "tissue",
            ],
        )

        assert result.exit_code == 0, result.output
        text = plain(result.output)
        assert HEADING.format(unit="tissue group", plural="groups") in text
        assert "NaiveMeanEffectsPredictor" in text.split("Metrics computed within")[1]
        assert loaded == [dataset_file]

    def test_the_resolved_dataset_replaces_the_names_handed_to_run(
        self, dataset_file: str, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = Recorder(return_value=ExperimentResult([make_run_result()]))
        monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: SimpleNamespace(name=name))
        patch_worker(monkeypatch, "drevalpy._run", "run", worker)
        out = ["-o", str(tmp_path / "results")]

        builtin = runner.invoke(app, ["run", "ElasticNet", "-d", dataset_file, *out, "--group-by", "drug"])
        annotated = runner.invoke(app, ["run", "ElasticNet", "-d", dataset_file, *out, "--group-by", "tissue"])

        assert builtin.exit_code == annotated.exit_code == 0, annotated.output
        assert worker.calls[0][1]["dataset"] == [dataset_file]
        assert isinstance(worker.calls[1][1]["dataset"], Dataset)

    def test_a_bad_annotation_fails_before_anything_runs(
        self, dataset_file: str, tmp_path: UPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worker = Recorder()
        monkeypatch.setattr("drevalpy.models.ModelSpec.from_name", lambda name: SimpleNamespace(name=name))
        patch_worker(monkeypatch, "drevalpy._run", "run", worker)

        result = runner.invoke(
            app, ["run", "ElasticNet", "-d", dataset_file, "-o", str(tmp_path / "results"), "--group-by", "nope"]
        )

        assert result.exit_code == 2
        assert "Unknown grouping 'nope'" in " ".join(plain(result.output).split())
        assert worker.call_count == 0
        assert not (tmp_path / "results").exists()


class TestHeading:
    @pytest.mark.parametrize(
        ("group_by", "unit", "plural"),
        [("drug", "drug", "drugs"), ("sample", "sample", "samples"), ("study", "study", "studies")],
    )
    def test_builtin_groupings_keep_their_wording(self, group_by: str, unit: str, plural: str) -> None:
        experiment = ExperimentResult([make_run_result()])

        heading = format_grouped_summary(experiment, group_by).splitlines()[0]

        assert heading == HEADING.format(unit=unit, plural=plural)

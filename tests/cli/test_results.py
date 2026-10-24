"""Tests for the commands that handle results and curation: ``aggregate``, ``report`` and ``curate``."""

from __future__ import annotations

import json
import logging

import anndata
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.cli.main import app
from drevalpy.curation import FIT_SPEEDS
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import HELP_ENV, Recorder, make_runner, patch_worker, plain
from tests.curation.conftest import build_dose_response_df
from tests.synthetic import DEFAULT_DATASET_NAME, make_experiment_result, make_run_result

runner = make_runner()


@pytest.fixture()
def run_files(tmp_path: UPath) -> list[UPath]:
    paths: list[UPath] = []
    for model_name in ("ElasticNet", "RandomForest"):
        for fold_index in range(2):
            path = tmp_path / f"{model_name}_fold_{fold_index}.npz"
            make_run_result(model_name=model_name, fold_index=fold_index).save(path)
            paths.append(path)
    return paths


def _invoke(run_files: list[UPath], out_dir: UPath, *extra: str):
    return runner.invoke(app, ["aggregate", *[str(p) for p in run_files], "--output-dir", str(out_dir), *extra])


def test_missing_results_is_a_usage_error() -> None:
    assert runner.invoke(app, ["aggregate"], env=HELP_ENV).exit_code == 2


def test_nonexistent_result_file_is_reported_and_creates_no_output(tmp_path: UPath) -> None:
    result = _invoke([tmp_path / "absent.npz"], tmp_path / "out")

    assert result.exit_code == 1
    assert plain(result.output).startswith("Error: ")
    assert not (tmp_path / "out").exists()


def test_aggregates_runs_into_a_reloadable_experiment(run_files: list[UPath], tmp_path: UPath) -> None:
    out_dir = tmp_path / "nested" / "experiment"

    result = _invoke(run_files, out_dir)

    assert result.exit_code == 0, result.output
    text = plain(result.output)
    assert f"Aggregated 4 runs into ExperimentResult at {out_dir}" in text
    assert "ExperimentResult" in text
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert sorted(meta["models"]) == ["ElasticNet", "RandomForest"]
    assert meta["dataset_name"] == DEFAULT_DATASET_NAME
    experiment = ExperimentResult.load(out_dir)
    assert (experiment.n_models, experiment.max_folds) == (2, 2)


def test_accepts_a_single_run(tmp_path: UPath) -> None:
    path = tmp_path / "one.npz"
    make_run_result().save(path)

    assert _invoke([path], tmp_path / "out").exit_code == 0


def test_mismatched_dataset_names_are_rejected(tmp_path: UPath) -> None:
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    make_run_result(dataset_name="DatasetA").save(first)
    make_run_result(dataset_name="DatasetB").save(second)

    result = _invoke([first, second], tmp_path / "out")

    assert result.exit_code == 1
    assert "Error: All RunResults must share the same dataset_name" in plain(result.output)


@pytest.mark.parametrize("option", ["drug", "sample", "DRUG"])
def test_group_by_prints_the_grouped_summary(run_files: list[UPath], tmp_path: UPath, option: str) -> None:
    result = _invoke(run_files, tmp_path / "out", "--group-by", option)

    assert result.exit_code == 0, result.output
    text = plain(result.output)
    assert f"Metrics computed within each {option.lower()}" in text
    assert "ElasticNet" in text.split("Metrics computed within")[1]


def test_group_by_is_silent_by_default_and_rejects_unknown_groupings(run_files: list[UPath], tmp_path: UPath) -> None:
    assert "Metrics computed within" not in plain(_invoke(run_files, tmp_path / "out").output)
    assert _invoke(run_files, tmp_path / "out2", "--group-by", "tissue").exit_code == 2


@pytest.fixture()
def experiment_dir(tmp_path: UPath) -> UPath:
    path = tmp_path / "experiment"
    make_experiment_result(n_models=2, n_folds=2).save(path)
    return path


@pytest.fixture()
def worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    patch_worker(monkeypatch, "drevalpy.visualization.report", "create_report", recorder)
    return recorder


def test_missing_or_nonexistent_experiment_dir_never_reaches_the_worker(worker: Recorder, tmp_path: UPath) -> None:
    assert runner.invoke(app, ["report"], env=HELP_ENV).exit_code == 2
    assert runner.invoke(app, ["report", str(tmp_path / "absent")]).exit_code != 0
    assert worker.call_count == 0


def test_forwards_defaults(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir)])

    assert result.exit_code == 0, result.output
    experiment, output_dir = worker.args
    assert isinstance(experiment, ExperimentResult)
    assert output_dir == "report"
    assert worker.kwargs == {"title": "Drug Response Evaluation", "reference_model": None}


def test_forwards_short_options_and_echoes_the_output_directory(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir), "-o", "out_here", "-t", "My Title", "-r", "Naive"])

    assert worker.args[1] == "out_here"
    assert worker.kwargs == {"title": "My Title", "reference_model": "Naive"}
    assert "Report generated at out_here" in plain(result.output)


def test_the_removed_dataset_option_is_a_usage_error(worker: Recorder, experiment_dir: UPath) -> None:
    result = runner.invoke(app, ["report", str(experiment_dir), "--dataset", "ds.h5mu"])
    helped = runner.invoke(app, ["report", "--help"], env=HELP_ENV)

    assert result.exit_code == 2
    assert "--dataset" not in plain(helped.output)


def test_the_experiment_is_loaded_without_trials(
    worker: Recorder, experiment_dir: UPath, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []
    original = ExperimentResult.load

    def spy(directory, *, with_trials=True):
        seen.append(with_trials)
        return original(directory, with_trials=with_trials)

    monkeypatch.setattr(ExperimentResult, "load", staticmethod(spy))

    runner.invoke(app, ["report", str(experiment_dir)])

    assert seen == [False]


@pytest.fixture()
def dose_response_df() -> pd.DataFrame:
    return build_dose_response_df()


@pytest.fixture()
def input_csv(dose_response_df: pd.DataFrame, tmp_path: UPath) -> UPath:
    path = tmp_path / "input.csv"
    dose_response_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def curate_worker(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder(return_value=anndata.AnnData(X=np.zeros((1, 1), dtype=np.float32)))
    patch_worker(monkeypatch, "drevalpy.curation", "curate", recorder)
    return recorder


@pytest.mark.parametrize("name", ["input.csv", "input.parquet", "input.pq", "input.CSV"])
def test_supported_suffixes_are_read(
    curate_worker: Recorder, dose_response_df: pd.DataFrame, tmp_path: UPath, name: str
) -> None:
    path = tmp_path / name
    if name.lower().endswith("csv"):
        dose_response_df.to_csv(path, index=False)
    else:
        dose_response_df.to_parquet(path)

    result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")])

    assert result.exit_code == 0, result.output
    assert len(curate_worker.args[0]) == len(dose_response_df)


def test_unsupported_suffix_is_rejected_before_the_fitter(curate_worker: Recorder, tmp_path: UPath) -> None:
    path = tmp_path / "input.tsv"
    path.write_text("drug\tsample\n")

    result = runner.invoke(app, ["curate", str(path), str(tmp_path / "out.h5ad")], env=HELP_ENV)

    assert result.exit_code == 2
    assert "Unsupported file format: .tsv" in plain(result.output)
    assert curate_worker.call_count == 0


def test_fit_option_defaults_and_overrides(curate_worker: Recorder, input_csv: UPath, tmp_path: UPath) -> None:
    out = str(tmp_path / "out.h5ad")

    runner.invoke(app, ["curate", str(input_csv), out])
    assert curate_worker.kwargs == {"max_workers": 4, "normalize": False, "fit_speed": "exhaustive"}

    curate_worker.calls.clear()
    runner.invoke(app, ["curate", str(input_csv), out, "-c", "2", "--normalize", "--fit-speed", "fast"])
    assert curate_worker.kwargs == {"max_workers": 2, "normalize": True, "fit_speed": "fast"}


@pytest.mark.parametrize("speed", FIT_SPEEDS)
def test_every_library_fit_speed_is_accepted_case_insensitively(
    curate_worker: Recorder, input_csv: UPath, tmp_path: UPath, speed: str
) -> None:
    runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), "--fit-speed", speed.upper()])

    assert curate_worker.kwargs["fit_speed"] == speed


@pytest.mark.parametrize(
    "extra", [["--fit-speed", "slow"], ["--fit-type", "OLS"], ["--cores", "all"]], ids=["speed", "fit-type", "cores"]
)
def test_invalid_options_are_usage_errors_before_the_fitter(
    curate_worker: Recorder, input_csv: UPath, tmp_path: UPath, extra: list[str]
) -> None:
    result = runner.invoke(app, ["curate", str(input_csv), str(tmp_path / "out.h5ad"), *extra])

    assert result.exit_code == 2
    assert curate_worker.call_count == 0


@pytest.mark.slow
def test_csv_to_h5ad_round_trip_with_the_real_fitter(tmp_path: UPath) -> None:
    input_path = tmp_path / "input.csv"
    build_dose_response_df().to_csv(input_path, index=False)
    output_path = tmp_path / "output.h5ad"
    logger = logging.getLogger("drevalpy")
    handlers, level = list(logger.handlers), logger.level
    try:
        result = runner.invoke(
            app, ["curate", str(input_path), str(output_path), "--cores", "1", "--fit-speed", "fast"]
        )
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)

    assert result.exit_code == 0, result.output
    adata = anndata.read_h5ad(output_path)
    assert adata.shape == (3, 2)
    assert {"EC50", "AUC"} <= set(adata.layers)

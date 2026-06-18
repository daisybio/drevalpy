"""Tests for the ``drevalpy curation`` CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from drevalpy.cli.main import app

runner = CliRunner()


def test_curation_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["curation", "--help"])
    assert result.exit_code == 0
    assert "split" in result.stdout
    assert "curvecurator" in result.stdout
    assert "combine" in result.stdout


def test_curation_root_reads_input_writes_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    output_file = tmp_path / "Toy.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})
    dataset = pd.DataFrame({"cell_line_name": ["S"]})

    def _fake_curate(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append({"args": args, "kwargs": kwargs})
        return dataset

    def _fake_write_dataset_csv(table: pd.DataFrame, path: Path) -> Path:
        _ = table
        calls.append({"output": path})
        return path

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr("drevalpy.cli.curation.curate", _fake_curate)
    monkeypatch.setattr("drevalpy.cli.curation.write_dataset_csv", _fake_write_dataset_csv)

    result = runner.invoke(
        app,
        [
            "curation",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert calls[0]["kwargs"]["input_filename"] == "Toy_raw.csv"
    assert calls[1]["output"] == output_file.resolve()


def test_curation_root_runs_synthetic_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    output_file = tmp_path / "Toy.csv"
    input_file.write_text(
        "dose,response,sample,drug\n" "1.0,0.95,A,D\n" "10.0,0.25,A,D\n",
        encoding="utf-8",
    )

    def _fake_run_pipeline_api(config, *, input_table, mad, device, gpu_chunk_size):
        _ = (config, mad, device, gpu_chunk_size)
        return pd.DataFrame(
            {
                "Name": input_table["Name"],
                "pEC50": [6.0],
                "Curve Slope": [1.0],
                "Curve Front": [1.0],
                "Curve Back": [0.1],
            }
        )

    monkeypatch.setattr("drevalpy.curation.fit._run_pipeline_api", _fake_run_pipeline_api)

    result = runner.invoke(
        app,
        [
            "curation",
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == str(output_file.resolve())
    assert pd.read_csv(output_file)["cell_line_name"].iloc[0] == "A"


def test_curation_split_reads_input_writes_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})
    artifact_dir = tmp_path / "work"

    def _fake_split(*args: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr("drevalpy.cli.curation.split", _fake_split)
    monkeypatch.setattr("drevalpy.cli.curation.write_split_artifacts", lambda result, output_dir: artifact_dir)

    result = runner.invoke(
        app,
        [
            "curation",
            "split",
            str(input_file),
            "--output-dir",
            str(tmp_path),
            "--gpu-min-curves",
            "25",
            "--gpu-chunk-size",
            "500",
        ],
    )
    assert result.exit_code == 0
    assert calls[0]["input_filename"] == "Toy_raw.csv"
    assert calls[0]["gpu_min_curves"] == 25
    assert calls[0]["gpu_chunk_size"] == 500
    assert calls[0]["gpu_available"] is False
    assert result.stdout.strip() == str(artifact_dir)


def test_curation_split_forwards_gpu_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})

    def _fake_split(*args: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr("drevalpy.cli.curation.split", _fake_split)
    monkeypatch.setattr(
        "drevalpy.cli.curation.write_split_artifacts",
        lambda result, output_dir: tmp_path / "work",
    )

    result = runner.invoke(
        app,
        [
            "curation",
            "split",
            str(input_file),
            "--output-dir",
            str(tmp_path),
            "--gpu-available",
        ],
    )
    assert result.exit_code == 0
    assert calls[0]["gpu_available"] is True


def test_curation_curvecurator_reads_input_writes_curves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "drug_treatment.json"
    config_file.write_text("{}", encoding="utf-8")
    input_file = tmp_path / "drug_treatment_input.parquet"
    input_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "drug_treatment.parquet"
    work_item = object()
    calls: list[tuple[object, Path]] = []

    def _fake_write_fit_curves(curves: pd.DataFrame, output_path: Path) -> Path:
        calls.append((curves, output_path))
        return output_path

    monkeypatch.setattr("drevalpy.cli.curation.read_work_item", lambda config_path, **kwargs: work_item)
    monkeypatch.setattr(
        "drevalpy.cli.curation.curvecurator",
        lambda item, **kwargs: type("FitResult", (), {"curves": pd.DataFrame({"pEC50": [1.0]})})(),
    )
    monkeypatch.setattr("drevalpy.cli.curation.write_fit_curves", _fake_write_fit_curves)

    result = runner.invoke(
        app,
        [
            "curation",
            "curvecurator",
            str(config_file),
            str(input_file),
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert calls[0][1] == output_file.resolve()
    assert result.stdout.strip() == str(output_file.resolve())


def test_curation_combine_reads_curve_files_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    curves_file = tmp_path / "drug_treatment.parquet"
    output = tmp_path / "Toy.csv"
    calls: list[dict[str, Any]] = []

    def _fake_combine(fit_results: list[Any]) -> pd.DataFrame:
        _ = fit_results
        calls.append({"combined": True})
        return pd.DataFrame()

    monkeypatch.setattr("drevalpy.cli.curation.resolve_curve_paths", lambda paths: [curves_file])
    monkeypatch.setattr("drevalpy.cli.curation.read_fit_results_from_paths", lambda paths: [])
    monkeypatch.setattr("drevalpy.cli.curation.combine", _fake_combine)
    monkeypatch.setattr("drevalpy.cli.curation.write_dataset_csv", lambda table, path: output)

    result = runner.invoke(
        app,
        [
            "curation",
            "combine",
            str(curves_file),
            "--output-file",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert calls[0]["combined"] is True
    assert result.stdout.strip() == str(output)

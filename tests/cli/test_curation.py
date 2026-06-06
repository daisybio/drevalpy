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
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})
    dataset = pd.DataFrame({"cell_line_name": ["S"]})

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr(
        "drevalpy.cli.curation.curate",
        lambda *args, **kwargs: calls.append({"args": args, "kwargs": kwargs}) or dataset,
    )
    monkeypatch.setattr(
        "drevalpy.cli.curation.write_dataset_csv",
        lambda table, path: calls.append({"output": path}) or path,
    )

    result = runner.invoke(
        app,
        [
            "curation",
            "--input-file",
            str(input_file),
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert calls[0]["kwargs"]["dataset_name"] == "Toy"
    assert calls[0]["kwargs"]["input_filename"] == "Toy_raw.csv"
    assert calls[1]["output"] == tmp_path / "Toy.csv"


def test_curation_root_runs_synthetic_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    output_dir = tmp_path / "out"
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

    monkeypatch.setattr("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", _fake_run_pipeline_api)

    result = runner.invoke(
        app,
        [
            "curation",
            "--input-file",
            str(input_file),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
    )

    output_file = output_dir / "Toy.csv"
    assert result.exit_code == 0
    assert result.stdout.strip() == str(output_file)
    assert pd.read_csv(output_file)["cell_line_name"].iloc[0] == "A"


def test_curation_split_reads_input_writes_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})
    manifest = tmp_path / "curation_manifest.json"

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr(
        "drevalpy.cli.curation.split",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setattr("drevalpy.cli.curation.write_split_artifacts", lambda result, output_dir: manifest)

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
    assert calls[0]["dataset_name"] == "Toy"
    assert calls[0]["input_filename"] == "Toy_raw.csv"
    assert calls[0]["gpu_min_curves"] == 25
    assert calls[0]["gpu_chunk_size"] == 500
    assert calls[0]["gpu_available"] is False
    assert result.stdout.strip() == str(manifest)


def test_curation_split_forwards_gpu_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = pd.DataFrame({"dose": [1.0], "response": [0.5], "sample": ["S"], "drug": ["D"]})

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr(
        "drevalpy.cli.curation.split",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        "drevalpy.cli.curation.write_split_artifacts",
        lambda result, output_dir: tmp_path / "curation_manifest.json",
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
    config_file = tmp_path / "Toy_drug_treatment_config.json"
    config_file.write_text("{}", encoding="utf-8")
    input_file = tmp_path / "Toy_drug_treatment_input.parquet"
    input_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "Toy_drug_treatment_curves.parquet"
    work_item = object()
    calls: list[tuple[object, Path]] = []

    monkeypatch.setattr("drevalpy.cli.curation.read_work_item", lambda config_path, **kwargs: work_item)
    monkeypatch.setattr(
        "drevalpy.cli.curation.curvecurator",
        lambda item, **kwargs: type("FitResult", (), {"curves": pd.DataFrame({"pEC50": [1.0]})})(),
    )
    monkeypatch.setattr(
        "drevalpy.cli.curation.write_fit_curves",
        lambda curves, output_path: calls.append((curves, output_path)) or output_path,
    )

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


def test_curation_combine_reads_manifest_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "curation_manifest.json"
    manifest.write_text('{"dataset_name":"Toy","input_filename":"Toy_raw.csv","work_items":[]}', encoding="utf-8")
    output = tmp_path / "Toy.csv"
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr("drevalpy.cli.curation.read_manifest", lambda path: {"dataset_name": "Toy"})
    monkeypatch.setattr("drevalpy.cli.curation.read_fit_results_from_manifest", lambda path: [])
    monkeypatch.setattr(
        "drevalpy.cli.curation.combine",
        lambda fit_results, dataset_name: calls.append({"dataset_name": dataset_name}) or pd.DataFrame(),
    )
    monkeypatch.setattr("drevalpy.cli.curation.write_dataset_csv", lambda table, path: output)

    result = runner.invoke(app, ["curation", "combine", str(manifest)])
    assert result.exit_code == 0
    assert calls[0]["dataset_name"] == "Toy"
    assert result.stdout.strip() == str(output)

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


def test_curation_root_calls_curate_to_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")

    def _fake_curate_to_csv(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return tmp_path / "Toy.csv"

    monkeypatch.setattr("drevalpy.cli.curation.curate_to_csv", _fake_curate_to_csv)

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
    assert calls[0]["dataset_name"] == "Toy"


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


def test_curation_split_calls_split_to_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    manifest = tmp_path / "curation_manifest.json"

    def _fake_split_to_disk(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return manifest

    monkeypatch.setattr("drevalpy.cli.curation.split_to_disk", _fake_split_to_disk)

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
    assert calls[0]["gpu_min_curves"] == 25
    assert calls[0]["gpu_chunk_size"] == 500
    assert calls[0]["gpu_available"] is False
    assert result.stdout.strip() == str(manifest)


def test_curation_split_forwards_gpu_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")

    monkeypatch.setattr(
        "drevalpy.cli.curation.split_to_disk",
        lambda **kwargs: calls.append(kwargs) or tmp_path / "curation_manifest.json",
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


def test_curation_curvecurator_calls_curvecurator_to_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "Toy_drug_treatment_config.json"
    config_file.write_text("{}", encoding="utf-8")
    input_file = tmp_path / "Toy_drug_treatment_input.parquet"
    input_file.write_text("", encoding="utf-8")
    output_file = tmp_path / "Toy_drug_treatment_curves.parquet"
    calls: list[tuple[Path, Path, Path]] = []

    def _fake_curvecurator_to_disk(config_path: Path, input_path: Path, output_path: Path, **kwargs: Any) -> Path:
        _ = kwargs
        calls.append((config_path, input_path, output_path))
        return output_path

    monkeypatch.setattr("drevalpy.cli.curation.curvecurator_to_disk", _fake_curvecurator_to_disk)

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
    assert calls[0] == (config_file.resolve(), input_file.resolve(), output_file.resolve())
    assert result.stdout.strip() == str(output_file.resolve())


def test_curation_combine_calls_combine_manifest_to_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = tmp_path / "curation_manifest.json"
    manifest.write_text('{"dataset_name":"Toy","input_filename":"Toy_raw.csv","work_items":[]}', encoding="utf-8")
    output = tmp_path / "Toy.csv"

    monkeypatch.setattr(
        "drevalpy.cli.curation.combine_manifest_to_csv",
        lambda manifest_path, output_file=None: output,
    )

    result = runner.invoke(app, ["curation", "combine", str(manifest)])
    assert result.exit_code == 0
    assert result.stdout.strip() == str(output)

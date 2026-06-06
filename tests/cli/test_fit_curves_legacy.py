"""Tests for legacy ``drevalpy-fit-curves`` argv forwarding."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from drevalpy.cli._fit_curves_legacy import dataset_name_from_input, forward_fit_curves_argv
from drevalpy.cli.legacy import fit_curves_cmd


def test_dataset_name_from_input_removes_raw_suffix() -> None:
    assert dataset_name_from_input(Path("MyDataset_raw.csv")) == "MyDataset"
    assert dataset_name_from_input(Path("MyDataset.csv")) == "MyDataset"


def test_forward_fit_curves_argv_applies_defaults(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    forwarded = forward_fit_curves_argv([str(input_file)])
    assert forwarded[0] == "curation"
    assert forwarded[1:3] == ["--input-file", str(input_file.resolve())]
    assert forwarded[3:5] == ["--output-dir", str(tmp_path.resolve())]
    assert forwarded[5:7] == ["--dataset-name", "Toy"]


def test_forward_fit_curves_argv_preserves_overrides() -> None:
    forwarded = forward_fit_curves_argv(
        [
            "/tmp/raw.csv",
            "--output-dir",
            "/tmp/out",
            "--dataset-name",
            "Custom",
            "--cores",
            "3",
            "--normalize",
            "--device",
            "cpu",
            "--chunk-size",
            "10",
            "--gpu-min-curves",
            "20",
            "--gpu-chunk-size",
            "30",
        ]
    )
    assert "--output-dir" in forwarded and "/tmp/out" in forwarded
    assert "--dataset-name" in forwarded and "Custom" in forwarded
    assert "--cores" in forwarded and "3" in forwarded
    assert "--normalize" in forwarded
    assert "--device" in forwarded and "cpu" in forwarded


def test_forward_fit_curves_argv_forwards_gpu_available_flag() -> None:
    forwarded = forward_fit_curves_argv(["/tmp/raw.csv", "--gpu-available"])
    assert "--gpu-available" in forwarded


def test_fit_curves_command_runs_in_memory_curation_with_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    raw_df = object()
    dataset = object()

    def _fake_curate(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return dataset

    def _fake_write_dataset_csv(table: object, path: Path) -> Path:
        _ = table
        calls.append({"output": path})
        return path

    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: raw_df)
    monkeypatch.setattr("drevalpy.cli.curation.curate", _fake_curate)
    monkeypatch.setattr("drevalpy.cli.curation.write_dataset_csv", _fake_write_dataset_csv)
    monkeypatch.setattr(sys, "argv", ["drevalpy-fit-curves", str(input_file)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(SystemExit) as exc_info:
            fit_curves_cmd()
    assert exc_info.value.code == 0

    assert calls[0]["kwargs"] == {
        "dataset_name": "Toy",
        "input_filename": "Toy_raw.csv",
        "cores": 1,
        "normalize": False,
        "device": "auto",
        "chunk_size": 1_000,
        "gpu_min_curves": 1_000,
        "gpu_chunk_size": 50_000,
        "gpu_available": False,
    }
    assert calls[1]["output"] == tmp_path.resolve() / "Toy.csv"
    assert capsys.readouterr().out.strip() == str(tmp_path.resolve() / "Toy.csv")


def test_fit_curves_command_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")
    monkeypatch.setattr("drevalpy.cli.curation.load_raw_curve_df", lambda path: object())
    monkeypatch.setattr("drevalpy.cli.curation.curate", lambda *args, **kwargs: object())
    monkeypatch.setattr("drevalpy.cli.curation.write_dataset_csv", lambda table, path: tmp_path / "Toy.csv")
    monkeypatch.setattr(sys, "argv", ["drevalpy-fit-curves", str(input_file)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit):
            fit_curves_cmd()

    assert any(issubclass(w.category, FutureWarning) and "drevalpy curation" in str(w.message) for w in caught)

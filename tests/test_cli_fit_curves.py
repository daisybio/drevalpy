"""Tests for the ``drevalpy-fit-curves`` command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from drevalpy import cli_fit_curves


def test_dataset_name_from_input_removes_raw_suffix() -> None:
    assert cli_fit_curves._dataset_name_from_input(Path("MyDataset_raw.csv")) == "MyDataset"
    assert cli_fit_curves._dataset_name_from_input(Path("MyDataset.csv")) == "MyDataset"


def test_fit_curves_command_calls_fit_curves_with_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")

    def _fake_fit_curves(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(cli_fit_curves, "fit_curves", _fake_fit_curves)
    monkeypatch.setattr(sys, "argv", ["drevalpy-fit-curves", str(input_file)])

    cli_fit_curves.fit_curves_cmd()

    assert calls == [
        {
            "input_file": str(input_file.resolve()),
            "output_dir": str(tmp_path.resolve()),
            "dataset_name": "Toy",
            "cores": 1,
            "normalize": False,
            "device": "auto",
            "chunk_size": 1_000,
            "gpu_min_curves": 1_000,
            "gpu_chunk_size": 50_000,
        }
    ]
    assert capsys.readouterr().out.strip() == str(tmp_path.resolve() / "Toy.csv")


def test_fit_curves_command_accepts_overrides(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    input_file = tmp_path / "raw.csv"
    out_dir = tmp_path / "out"
    input_file.write_text("dose,response,sample,drug\n1,0.5,S,D\n", encoding="utf-8")

    def _fake_fit_curves(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(cli_fit_curves, "fit_curves", _fake_fit_curves)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drevalpy-fit-curves",
            str(input_file),
            "--output_dir",
            str(out_dir),
            "--dataset_name",
            "Custom",
            "--cores",
            "3",
            "--normalize",
            "--device",
            "cpu",
            "--chunk_size",
            "10",
            "--gpu_min_curves",
            "20",
            "--gpu_chunk_size",
            "30",
        ],
    )

    cli_fit_curves.fit_curves_cmd()

    assert calls[0]["output_dir"] == str(out_dir.resolve())
    assert calls[0]["dataset_name"] == "Custom"
    assert calls[0]["cores"] == 3
    assert calls[0]["normalize"] is True
    assert calls[0]["device"] == "cpu"
    assert calls[0]["chunk_size"] == 10
    assert calls[0]["gpu_min_curves"] == 20
    assert calls[0]["gpu_chunk_size"] == 30

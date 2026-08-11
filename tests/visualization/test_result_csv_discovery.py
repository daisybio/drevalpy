"""Tests for result_csv_discovery."""

from __future__ import annotations

from pathlib import Path

from drevalpy.visualization._legacy.result_csv_discovery import discover_result_csv_files


def test_discover_result_csv_files_delegates_layout(tmp_path: Path) -> None:
    pred = tmp_path / "GDSC1" / "LCO" / "ElasticNet" / "predictions" / "predictions_split_0.csv"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.write_text("cell_line_name,pubchem_id,response,predictions\n", encoding="utf-8")
    assert discover_result_csv_files(tmp_path, "GDSC1") == [pred]

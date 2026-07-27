"""Tests for cellosaurus_parse streaming parser."""

from __future__ import annotations

from pathlib import Path

from drevalpy.datasets.cellosaurus_parse import parse_cellosaurus


def test_parse_cellosaurus_maps_cvcl_ids(tmp_path: Path) -> None:
    text = (
        "ID   Test line\n"
        "AC   CVCL_TEST1;\n"
        "CC   Derived from site: In culture; Lung;\n"
        "DI   DOID; DOID:3908; lung adenocarcinoma;\n"
        "//\n"
    )
    path = tmp_path / "cellosaurus.txt"
    path.write_text(text, encoding="utf-8")
    id_to_name, id_to_site, id_to_disease = parse_cellosaurus(path)
    assert id_to_name["CVCL_TEST1"] == "Test line"
    assert id_to_site["CVCL_TEST1"] == "Lung"
    assert id_to_disease["CVCL_TEST1"] == "lung adenocarcinoma"

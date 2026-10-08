"""Tests for drevalpy.visualization.utils."""

from __future__ import annotations

from pathlib import Path

from drevalpy.datasets.splits import SplitParams, write_split_manifest
from drevalpy.visualization.utils import _discover_result_csv_files, _resolve_result_test_mode


def test_discover_result_csv_files_finds_custom_split_label_results(tmp_path: Path) -> None:
    """
    Discover result CSVs under arbitrary split-label directories.

    :param tmp_path: Temporary path provided by pytest.
    """
    pred = tmp_path / "GDSC1" / "scaling-lco" / "ElasticNet" / "predictions" / "predictions_split_0.csv"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.write_text("cell_line_name,pubchem_id,response,predictions\n", encoding="utf-8")

    discovered = _discover_result_csv_files(tmp_path, "GDSC1")
    assert discovered == [pred]


def test_discover_result_csv_files_skips_split_role_csvs(tmp_path: Path) -> None:
    """
    Ignore split role CSVs stored under ``splits/``.

    :param tmp_path: Temporary path provided by pytest.
    """
    split_csv = tmp_path / "GDSC1" / "LCO" / "splits" / "cv_split_0_train.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split_csv.write_text("cell_line_name,pubchem_id,response\n", encoding="utf-8")

    assert _discover_result_csv_files(tmp_path, "GDSC1") == []


def test_discover_result_csv_files_includes_all_result_categories(tmp_path: Path) -> None:
    """
    Collect CSVs from predictions, cross_study, randomization, and robustness folders.

    :param tmp_path: Temporary path provided by pytest.
    """
    categories = {
        "predictions": "predictions_split_0.csv",
        "cross_study": "cross_study_GDSC2_split_0.csv",
        "randomization": "randomization_SVCC_split_0.csv",
        "robustness": "robustness_1_split_0.csv",
    }
    created: list[Path] = []
    for category, filename in categories.items():
        path = tmp_path / "GDSC1" / "LCO" / "ElasticNet" / category / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cell_line_name,pubchem_id,response,predictions\n", encoding="utf-8")
        created.append(path)

    discovered = _discover_result_csv_files(tmp_path, "GDSC1")
    assert discovered == created


def test_resolve_result_test_mode_uses_manifest(tmp_path: Path) -> None:
    """
    Resolve semantic test mode from split manifests for custom result labels.

    :param tmp_path: Temporary path provided by pytest.
    """
    split_dir = tmp_path / "GDSC1" / "scaling-lco" / "splits"
    params = SplitParams(
        test_mode="LCO",
        n_cv_splits=1,
        validation_ratio=0.1,
        random_state=42,
        split_early_stopping=True,
    )
    write_split_manifest(split_dir, params=params, split_label="scaling-lco", splits=[{"split_index": 0}])
    assert _resolve_result_test_mode(tmp_path, "GDSC1", "scaling-lco") == "LCO"


def test_resolve_result_test_mode_falls_back_to_split_label(tmp_path: Path) -> None:
    """
    Fall back to the result directory label when no manifest exists.

    :param tmp_path: Temporary path provided by pytest.
    """
    assert _resolve_result_test_mode(tmp_path, "GDSC1", "LCO") == "LCO"

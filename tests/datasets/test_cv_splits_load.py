"""Tests for cv_splits_load."""

from __future__ import annotations

from pathlib import Path

from drevalpy.datasets.cv_splits_load import load_cv_splits_from_dir


def test_load_cv_splits_from_dir_train_test(tmp_path: Path) -> None:
    header = "cell_line_name,pubchem_id,response\n"
    (tmp_path / "cv_split_0_train.csv").write_text(header + "a,d,1\n")
    (tmp_path / "cv_split_0_test.csv").write_text(header + "b,d,2\n")
    splits = load_cv_splits_from_dir(str(tmp_path), "testset")
    assert len(splits) == 1
    assert len(splits[0]["train"]) == 1
    assert len(splits[0]["test"]) == 1

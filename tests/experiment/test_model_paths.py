"""Tests for experiment model path helpers."""

from __future__ import annotations

from pathlib import Path

from drevalpy.experiment.model_paths import (
    generate_data_saving_path,
    generate_final_model_checkpoint_path,
)


def test_generate_data_saving_path_creates_directory(tmp_path: Path) -> None:
    path = generate_data_saving_path("ElasticNet", None, str(tmp_path), "predictions")
    assert Path(path).is_dir()
    assert Path(path).name == "predictions"


def test_generate_final_model_checkpoint_path_is_file_stem(tmp_path: Path) -> None:
    path = generate_final_model_checkpoint_path("ElasticNet", None, str(tmp_path))
    assert Path(path).parent.is_dir()
    assert not Path(path).exists()
    assert Path(path).name == "final_model"

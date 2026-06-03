"""Tests for in-process CurveCurator runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from drevalpy.datasets.curvecurator import _prepare_raw_data
from drevalpy.datasets.curvecurator_runner import (
    CurveCuratorWorkItem,
    _fit_with_fallback,
    finalize_config,
    run_curvecurator_work_items,
)


def _minimal_config() -> dict:
    return {
        "Meta": {"id": "data.csv"},
        "Experiment": {"experiments": [0, 1], "doses": [0.0, 1.0], "dose_scale": "1e-06"},
        "Paths": {
            "input_file": "curvecurator_input.tsv",
            "curves_file": "curves.tsv",
        },
        "F Statistic": {"alpha": 0.05, "fc_lim": 0.45},
    }


def test_finalize_config_records_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "job_config.json"
    config = finalize_config(_minimal_config(), config_path=config_path)
    assert config["__file__"]["Path"] == str(config_path.resolve())


def test_fit_with_fallback_retries_on_oom() -> None:
    fitted = pd.DataFrame({"Name": ["A|D"], "pEC50": [6.0]})
    input_table = pd.DataFrame({"Name": ["A|D"], "Raw 0": [1.0]})
    calls: list[str] = []

    def _fake_run(config, *, input_table, mad, device, gpu_chunk_size):
        _ = (config, input_table, mad, gpu_chunk_size)
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return fitted

    with patch("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", side_effect=_fake_run):
        result = _fit_with_fallback(
            _minimal_config(),
            input_table,
            "cuda",
            50_000,
            "chunk_0",
            mad=False,
        )

    assert calls == ["cuda", "cpu"]
    assert len(result) == 1


def test_run_curvecurator_work_items_writes_curves_tsv(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "group"
    chunk_dir.mkdir()
    (chunk_dir / "curvecurator_input.tsv").write_text("Name\tRaw 0\nA|D\t1.0\n", encoding="utf-8")
    fitted = pd.DataFrame(
        {
            "Name": ["A|D"],
            "pEC50": [6.0],
            "Curve Slope": [1.0],
            "Curve Front": [1.0],
            "Curve Back": [0.1],
        }
    )
    item = CurveCuratorWorkItem(chunk_dir=chunk_dir, config=_minimal_config(), n_curves=1)

    with patch("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", return_value=fitted) as mock_api:
        run_curvecurator_work_items(
            [item],
            cores=1,
            device="cpu",
            gpu_min_curves=10**9,
            gpu_chunk_size=50_000,
            mad=False,
        )
        mock_api.assert_called_once()
        assert mock_api.call_args.kwargs["device"] == "cpu"

    curves_path = chunk_dir / "curves.tsv"
    assert curves_path.is_file()
    assert "pEC50" in curves_path.read_text(encoding="utf-8")


def test_run_curvecurator_work_items_passes_gpu_chunk_size(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "group"
    chunk_dir.mkdir()
    (chunk_dir / "curvecurator_input.tsv").write_text("Name\tRaw 0\n", encoding="utf-8")
    item = CurveCuratorWorkItem(chunk_dir=chunk_dir, config=_minimal_config(), n_curves=5000)
    fitted = pd.DataFrame({"Name": ["A|D"], "pEC50": [6.0]})

    with patch("drevalpy.curation._curvecurator.device.resolve_device", return_value="cuda"):
        with patch("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", return_value=fitted) as mock_api:
            run_curvecurator_work_items(
                [item],
                cores=1,
                device="auto",
                gpu_min_curves=1000,
                gpu_chunk_size=12_345,
                mad=False,
            )
            assert mock_api.call_args.kwargs["device"] == "cuda"
            assert mock_api.call_args.kwargs["gpu_chunk_size"] == 12_345


def test_run_curvecurator_work_items_raises_on_failure(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "group"
    chunk_dir.mkdir()
    item = CurveCuratorWorkItem(chunk_dir=chunk_dir, config=_minimal_config(), n_curves=1)

    with patch(
        "drevalpy.datasets.curvecurator_runner._run_one_work_item",
        side_effect=RuntimeError("fit failed"),
    ):
        with pytest.raises(RuntimeError, match="1 CurveCurator fit\\(s\\) failed"):
            run_curvecurator_work_items([item], cores=1, device="cpu", gpu_min_curves=10**9)


def test_prepare_raw_data_without_replicate_column(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample": ["A", "A"],
            "drug": ["D", "D"],
            "dose": [1.0, 10.0],
            "response": [0.9, 0.1],
        }
    )
    n_exp, doses, n_replicates, n_curves = _prepare_raw_data(df, tmp_path, "group")

    assert n_exp == 3
    assert doses == [0.0, 1.0, 10.0]
    assert n_replicates == 1
    assert n_curves == 1
    assert (tmp_path / "group" / "curvecurator_input.tsv").is_file()


def test_prepare_raw_data_pools_replicates_into_one_curve(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample": ["A", "A", "A", "A"],
            "drug": ["D", "D", "D", "D"],
            "dose": [1.0, 10.0, 1.0, 10.0],
            "response": [0.9, 0.1, 0.8, 0.2],
            "replicate": [0, 0, 1, 1],
        }
    )
    n_exp, doses, n_replicates, n_curves = _prepare_raw_data(df, tmp_path, "group")

    assert n_exp == 6
    assert doses == [0.0, 0.0, 1.0, 1.0, 10.0, 10.0]
    assert n_replicates == 2
    assert n_curves == 1

"""Tests for in-process CurveCurator execution."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from drevalpy.curation._curvecurator.curvecurator import _fit_with_fallback, _fit_work_item, finalize_config
from drevalpy.curation._curvecurator.types import CurationWorkItem


def _minimal_config() -> dict:
    return {
        "Meta": {"id": "Toy_raw.csv", "description": "Toy", "condition": "drug_treatment"},
        "Experiment": {"experiments": [0, 1], "doses": [0.0, 1.0], "dose_scale": "1e-06"},
        "Paths": {
            "input_file": "curvecurator_input.tsv",
            "curves_file": "curves.tsv",
        },
        "F Statistic": {"alpha": 0.05, "fc_lim": 0.45},
    }


def _gpu_routed_work_item() -> CurationWorkItem:
    return CurationWorkItem(
        work_id="gpu_job",
        dataset_name="Toy",
        group_key="group",
        chunk_index=None,
        input_table=pd.DataFrame({"Name": ["A|D"], "Raw 0": [1.0], "Raw 1": [0.5]}),
        config={
            "Meta": {"id": "Toy_raw.csv", "description": "Toy", "condition": "group"},
            "Paths": {"input_file": "curvecurator_input.tsv", "curves_file": "curves.tsv"},
            "Routing": {"n_curves": 1, "device": "auto"},
        },
        n_curves=1,
        input_filename="Toy_raw.csv",
    )


def test_fit_work_item_warns_when_gpu_routed_job_resolves_to_cpu(tmp_path: Path) -> None:
    fitted = pd.DataFrame({"Name": ["A|D"], "pEC50": [6.0]})

    with patch("drevalpy.curation._curvecurator.curvecurator.effective_device", return_value="cpu"):
        with patch(
            "drevalpy.curation._curvecurator.curvecurator._fit_with_fallback",
            return_value=fitted,
        ) as mock_fit:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _fit_work_item(
                    _gpu_routed_work_item(),
                    device="auto",
                    gpu_min_curves=1_000,
                    gpu_chunk_size=50_000,
                    mad=False,
                )

    assert len(result) == 1
    mock_fit.assert_called_once()
    assert mock_fit.call_args.args[2] == "cpu"
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "GPU-routed job gpu_job resolved to CPU on this node" in str(w.message)
        for w in caught
    )


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


def test_run_pipeline_api_passes_in_memory_table_to_fork() -> None:
    input_table = pd.DataFrame({"Name": ["A|D"], "Raw 0": [1.0], "Raw 1": [0.5]})
    config = _minimal_config()
    captured: dict[str, object] = {}

    def _fake_fork_api(cfg, data, *, mad, device, gpu_chunk_size):
        captured["config"] = cfg
        captured["data"] = data
        captured["kwargs"] = {"mad": mad, "device": device, "gpu_chunk_size": gpu_chunk_size}
        return pd.DataFrame({"Name": data["Name"], "pEC50": [6.0]})

    with patch("drevalpy.curation._curvecurator.curvecurator.run_pipeline_api", side_effect=_fake_fork_api):
        from drevalpy.curation._curvecurator.curvecurator import _run_pipeline_api

        result = _run_pipeline_api(
            config,
            input_table=input_table,
            mad=False,
            device="cpu",
            gpu_chunk_size=50_000,
        )

    pd.testing.assert_frame_equal(captured["data"], input_table)
    assert captured["kwargs"] == {"mad": False, "device": "cpu", "gpu_chunk_size": 50_000}
    assert len(result) == 1


def test_finalize_config_preserves_input_file_and_resolves_output_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "job_config.json"
    config = {
        "Paths": {
            "input_file": "curvecurator_input.tsv",
            "curves_file": "curves.tsv",
            "mad_file": "mad.txt",
        }
    }

    finalized = finalize_config(config, config_path=config_path)

    assert finalized["Paths"]["input_file"] == "curvecurator_input.tsv"
    assert finalized["Paths"]["curves_file"] == str((tmp_path / "curves.tsv").resolve())
    assert finalized["Paths"]["mad_file"] == str((tmp_path / "mad.txt").resolve())
    assert finalized["__file__"] == {"Path": str(config_path.resolve())}

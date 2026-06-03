"""Tests for in-process CurveCurator execution."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from drevalpy.curation._curvecurator.curvecurator import _fit_work_item
from drevalpy.curation._curvecurator.types import CurationWorkItem


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

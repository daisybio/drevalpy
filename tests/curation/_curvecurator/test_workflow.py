"""Synthetic end-to-end tests for the CurveCurator curation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drevalpy.curation import curate, load_raw_curve_df, split, write_dataset_csv
from drevalpy.curation._curvecurator.combine import combine
from drevalpy.curation._curvecurator.curvecurator import curvecurator
from drevalpy.curation._curvecurator.io import (
    job_config_path,
    job_curves_path,
    job_input_path,
    read_fit_results_from_manifest,
    write_fit_curves,
    write_split_artifacts,
)


def _write_synthetic_raw(input_file: Path) -> None:
    input_file.write_text(
        "\n".join(
            [
                "dose,response,sample,drug,replicate",
                "1.0,0.95,A,D1,0",
                "10.0,0.25,A,D1,0",
                "1.0,0.90,A,D1,1",
                "10.0,0.20,A,D1,1",
                "1.0,0.85,B,D1,0",
                "10.0,0.30,B,D1,0",
                "1.0,0.80,B,D1,1",
                "10.0,0.35,B,D1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _synthetic_curves(names: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": names.to_list(),
            "pEC50": [6.0 + index for index in range(len(names))],
            "pEC50 Error": [0.1] * len(names),
            "Curve Slope": [1.0] * len(names),
            "Curve Front": [1.0] * len(names),
            "Curve Back": [0.1] * len(names),
            "Curve Fold Change": [10.0] * len(names),
            "Curve AUC": [0.5] * len(names),
            "Curve R2": [0.9] * len(names),
            "Curve P_Value": [0.01] * len(names),
            "Curve Relevance Score": [1.0] * len(names),
            "Curve F_Value": [10.0] * len(names),
            "Curve Log P_Value": [2.0] * len(names),
            "Signal Quality": [1.0] * len(names),
            "Curve RMSE": [0.01] * len(names),
            "Curve F_Value SAM Corrected": [9.0] * len(names),
            "Curve Regulation": ["down"] * len(names),
        }
    )


def _fake_run_pipeline_api(config, *, input_table, mad, device, gpu_chunk_size):
    _ = (config, mad, device, gpu_chunk_size)
    return _synthetic_curves(input_table["Name"])


def test_curate_runs_synthetic_in_process_workflow(tmp_path: Path, monkeypatch) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    output_dir = tmp_path / "out"
    _write_synthetic_raw(input_file)
    monkeypatch.setattr("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", _fake_run_pipeline_api)

    raw_df = load_raw_curve_df(input_file)
    dataset = curate(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1, device="cpu")
    output_path = write_dataset_csv(dataset, output_dir / "Toy.csv")

    dataset_from_disk = pd.read_csv(output_path)
    assert output_path == output_dir / "Toy.csv"
    assert set(dataset_from_disk["cell_line_name"]) == {"A", "B"}
    assert set(dataset_from_disk["pubchem_id"]) == {"D1"}
    assert {"EC50_curvecurator", "IC50_curvecurator", "LN_IC50_curvecurator"}.issubset(dataset_from_disk.columns)


def test_stepwise_transport_workflow_runs_synthetic_jobs(tmp_path: Path, monkeypatch) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    work_dir = tmp_path / "work"
    output_file = tmp_path / "Toy.csv"
    _write_synthetic_raw(input_file)
    monkeypatch.setattr("drevalpy.curation._curvecurator.curvecurator._run_pipeline_api", _fake_run_pipeline_api)

    raw_df = load_raw_curve_df(input_file)
    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)
    manifest_path = write_split_artifacts(split_result, work_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for entry in manifest["work_items"]:
        job_id = entry["job_id"]
        work_item = next(item for item in split_result.work_items if item.work_id == job_id)
        fit_result = curvecurator(work_item, device="cpu")
        written_curves = write_fit_curves(fit_result.curves, job_curves_path(work_dir, job_id))
        assert written_curves == job_curves_path(work_dir, job_id)
        assert job_config_path(work_dir, job_id).is_file()
        assert job_input_path(work_dir, job_id).is_file()

    fit_results = read_fit_results_from_manifest(manifest_path)
    dataset = combine(fit_results, dataset_name=manifest["dataset_name"])
    combined_path = write_dataset_csv(dataset, output_file)

    dataset_from_disk = pd.read_csv(combined_path)
    assert combined_path == output_file
    assert set(dataset_from_disk["cell_line_name"]) == {"A", "B"}

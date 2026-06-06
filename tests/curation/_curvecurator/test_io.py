"""Tests for curation CLI transport serialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drevalpy.curation import curvecurator, load_raw_curve_df, split
from drevalpy.curation._curvecurator.io import (
    CONFIG_SUFFIX,
    CURVES_SUFFIX,
    INPUT_SUFFIX,
    MANIFEST_FILENAME,
    job_config_path,
    job_curves_path,
    job_input_path,
    read_fit_results_from_manifest,
    read_work_item,
    write_fit_curves,
    write_split_artifacts,
)


def test_write_and_read_split_artifacts_flat_layout(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "work"
    raw_df = load_raw_curve_df(input_file)
    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)
    manifest_path = write_split_artifacts(split_result, output_dir)

    assert manifest_path.name == MANIFEST_FILENAME
    job_id = split_result.work_items[0].work_id
    assert job_config_path(output_dir, job_id).is_file()
    assert job_input_path(output_dir, job_id).is_file()
    assert not (output_dir / job_id).exists()

    work_item = read_work_item(job_config_path(output_dir, job_id))
    assert work_item.work_id == job_id
    assert list(work_item.input_table.columns)[0] == "Name"
    assert work_item.config["Meta"]["description"] == "Toy"

    payload = job_config_path(output_dir, job_id).read_text(encoding="utf-8")
    assert CONFIG_SUFFIX in job_config_path(output_dir, job_id).name
    assert INPUT_SUFFIX in job_input_path(output_dir, job_id).name
    assert '"Routing"' in payload
    assert '"n_curves"' in payload
    assert '"device"' in payload
    assert '"Meta"' in payload


def test_read_fit_results_from_manifest_requires_curves(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "work"
    raw_df = load_raw_curve_df(input_file)
    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)
    manifest_path = write_split_artifacts(split_result, output_dir)

    with pytest.raises(FileNotFoundError, match="Missing fitted curves file"):
        read_fit_results_from_manifest(manifest_path)

    job_id = split_result.work_items[0].work_id
    pd.DataFrame({"Name": ["A|D"], "pEC50": [6.0]}).to_parquet(job_curves_path(output_dir, job_id), index=False)
    fit_results = read_fit_results_from_manifest(manifest_path)
    assert len(fit_results) == 1
    assert fit_results[0].work_id == job_id
    assert CURVES_SUFFIX in job_curves_path(output_dir, job_id).name


def test_read_work_item_accepts_explicit_input_path(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "work"
    raw_df = load_raw_curve_df(input_file)
    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)
    write_split_artifacts(split_result, output_dir)

    job_id = split_result.work_items[0].work_id
    config_path = job_config_path(output_dir, job_id)
    custom_input = tmp_path / "custom_input.parquet"
    job_input_path(output_dir, job_id).replace(custom_input)

    work_item = read_work_item(config_path, input_path=custom_input)
    assert work_item.work_id == job_id
    assert list(work_item.input_table.columns)[0] == "Name"


def test_write_fit_curves_writes_explicit_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    raw_df = load_raw_curve_df(input_file)
    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)
    work_item = split_result.work_items[0]
    output_path = tmp_path / "fitted" / "custom_curves.parquet"
    fitted = pd.DataFrame({"Name": ["A|D"], "pEC50": [6.0]})

    monkeypatch.setattr(
        "drevalpy.curation._curvecurator.curvecurator._run_pipeline_api",
        lambda *args, **kwargs: fitted,
    )

    fit_result = curvecurator(work_item, device="cpu")
    written = write_fit_curves(fit_result.curves, output_path)
    assert written == output_path.resolve()
    assert output_path.is_file()
    assert pd.read_parquet(output_path)["pEC50"].iloc[0] == 6.0

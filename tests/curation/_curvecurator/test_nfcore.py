"""Tests for nf-core viability preprocess/postprocess hooks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from drevalpy.curation._curvecurator.nfcore import run_postprocess_viability, run_preprocess_raw_viability


def _write_synthetic_raw(input_file: Path) -> None:
    input_file.write_text(
        "\n".join(
            [
                "dose,response,sample,drug,replicate",
                "1.0,0.95,A,D1,0",
                "10.0,0.25,A,D1,0",
                "1.0,0.85,B,D1,0",
                "10.0,0.30,B,D1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _legacy_curves_tsv(path: Path) -> None:
    pd.DataFrame(
        {
            "Name": ["A|D1", "B|D1"],
            "pEC50": [6.0, 6.5],
            "pEC50 Error": [0.1, 0.1],
            "Curve Slope": [1.0, 1.0],
            "Curve Front": [1.0, 1.0],
            "Curve Back": [0.1, 0.1],
            "Curve Fold Change": [10.0, 10.0],
            "Curve AUC": [0.5, 0.5],
            "Curve R2": [0.9, 0.9],
            "Curve P_Value": [0.01, 0.01],
            "Curve Relevance Score": [1.0, 1.0],
            "Curve F_Value": [10.0, 10.0],
            "Curve Log P_Value": [2.0, 2.0],
            "Signal Quality": [1.0, 1.0],
            "Curve RMSE": [0.01, 0.01],
            "Curve F_Value SAM Corrected": [9.0, 9.0],
            "Curve Regulation": ["down", "down"],
        }
    ).to_csv(path, sep="\t", index=False)


def test_run_preprocess_raw_viability_calls_split_to_disk(tmp_path: Path) -> None:
    dataset_name = "Toy"
    dataset_dir = tmp_path / "data" / dataset_name
    dataset_dir.mkdir(parents=True)
    _write_synthetic_raw(dataset_dir / f"{dataset_name}_raw.csv")

    with patch("drevalpy.curation._curvecurator.nfcore.split_to_disk") as mock_split:
        run_preprocess_raw_viability(path_data=str(tmp_path / "data"), dataset_name=dataset_name, cores=2)

    mock_split.assert_called_once_with(
        input_file=dataset_dir / f"{dataset_name}_raw.csv",
        output_dir=dataset_dir,
        dataset_name=dataset_name,
        cores=2,
    )


def test_run_postprocess_viability_uses_manifest_when_present(tmp_path: Path) -> None:
    dataset_name = "Toy"
    dataset_dir = tmp_path / dataset_name
    dataset_dir.mkdir(parents=True)
    manifest = dataset_dir / "curation_manifest.json"
    manifest.write_text(
        json.dumps({"dataset_name": dataset_name, "input_filename": "Toy_raw.csv", "work_items": []}),
        encoding="utf-8",
    )

    with patch("drevalpy.curation._curvecurator.nfcore.combine_manifest_to_csv") as mock_combine:
        run_postprocess_viability(dataset_name=dataset_name, path_data=str(tmp_path))

    mock_combine.assert_called_once_with(manifest)


def test_run_postprocess_viability_falls_back_to_legacy_curves_tsv(tmp_path: Path) -> None:
    dataset_name = "Toy"
    dataset_dir = tmp_path / dataset_name / "group_a"
    dataset_dir.mkdir(parents=True)
    _legacy_curves_tsv(dataset_dir / "curves.tsv")

    run_postprocess_viability(dataset_name=dataset_name, path_data=str(tmp_path))

    output_csv = tmp_path / dataset_name / f"{dataset_name}.csv"
    assert output_csv.is_file()
    dataset = pd.read_csv(output_csv, index_col=0)
    assert set(dataset["cell_line_name"]) == {"A", "B"}
    assert set(dataset["pubchem_id"]) == {"D1"}

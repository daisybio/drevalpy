"""Tests for in-memory curation split/combine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.curation import combine, load_raw_curve_df, split
from drevalpy.curation._curvecurator.split import prepare_input_table
from drevalpy.curation._curvecurator.types import CurationFitResult, CurationWorkItem


def test_split_returns_in_memory_work_items(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    raw_df = load_raw_curve_df(input_file)

    split_result = split(raw_df, dataset_name="Toy", input_filename=input_file.name, cores=1)

    assert split_result.dataset_name == "Toy"
    assert len(split_result.work_items) == 1
    work_item = split_result.work_items[0]
    assert work_item.n_curves == 1
    assert "Name" in work_item.input_table.columns
    assert work_item.config["Meta"]["description"] == "Toy"
    assert work_item.config["Routing"] == {"n_curves": 1, "device": "cpu"}


def test_split_records_auto_routing_for_accelerator_sized_chunks() -> None:
    curve_df = pd.DataFrame(
        {
            "dose": [1.0, 10.0, 1.0, 10.0, 1.0, 10.0],
            "response": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "sample": ["A", "A", "B", "B", "C", "C"],
            "drug": ["D", "D", "D", "D", "D", "D"],
        }
    )

    split_result = split(
        curve_df,
        dataset_name="Toy",
        input_filename="Toy_raw.csv",
        device="auto",
        gpu_min_curves=2,
        gpu_available=True,
    )

    assert len(split_result.work_items) == 1
    assert split_result.work_items[0].config["Routing"] == {"n_curves": 3, "device": "auto"}


def test_split_defaults_to_cpu_chunking_without_gpu_available() -> None:
    curve_df = pd.DataFrame(
        {
            "dose": [1.0, 10.0, 1.0, 10.0, 1.0, 10.0],
            "response": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "sample": ["A", "A", "B", "B", "C", "C"],
            "drug": ["D", "D", "D", "D", "D", "D"],
        }
    )

    split_result = split(
        curve_df,
        dataset_name="Toy",
        input_filename="Toy_raw.csv",
        device="auto",
        gpu_min_curves=2,
        gpu_available=False,
    )

    assert len(split_result.work_items) == 1
    assert split_result.work_items[0].config["Routing"] == {"n_curves": 3, "device": "cpu"}


def test_split_keeps_small_chunks_on_cpu_when_gpu_available(tmp_path: Path) -> None:
    input_file = tmp_path / "Toy_raw.csv"
    input_file.write_text(
        "dose,response,sample,drug\n1.0,0.9,A,D\n10.0,0.1,A,D\n",
        encoding="utf-8",
    )
    raw_df = load_raw_curve_df(input_file)

    split_result = split(
        raw_df,
        dataset_name="Toy",
        input_filename=input_file.name,
        device="auto",
        gpu_min_curves=2,
        gpu_available=True,
    )

    assert split_result.work_items[0].config["Routing"] == {"n_curves": 1, "device": "cpu"}


def test_prepare_input_table_pools_replicates_into_one_curve() -> None:
    curve_df = pd.DataFrame(
        {
            "sample": ["A", "A", "A", "A"],
            "drug": ["D", "D", "D", "D"],
            "dose": [1.0, 10.0, 1.0, 10.0],
            "response": [0.9, 0.1, 0.8, 0.2],
            "replicate": [0, 0, 1, 1],
        }
    )

    input_table, n_exp, doses, n_replicates, n_curves = prepare_input_table(curve_df)

    assert list(input_table["Name"]) == ["A|D"]
    assert n_exp == 6
    assert doses == [0.0, 0.0, 1.0, 1.0, 10.0, 10.0]
    assert n_replicates == 2
    assert n_curves == 1


def test_combine_builds_dataset_table() -> None:
    curves = pd.DataFrame(
        {
            "Name": ["A|D"],
            "pEC50": [6.0],
            "pEC50 Error": [0.1],
            "Curve Slope": [1.0],
            "Curve Front": [1.0],
            "Curve Back": [0.1],
            "Curve Fold Change": [10.0],
            "Curve AUC": [0.5],
            "Curve R2": [0.9],
            "Curve P_Value": [0.01],
            "Curve Relevance Score": [1.0],
            "Curve F_Value": [10.0],
            "Curve Log P_Value": [2.0],
            "Signal Quality": [1.0],
            "Curve RMSE": [0.01],
            "Curve F_Value SAM Corrected": [9.0],
            "Curve Regulation": ["down"],
        }
    )
    fit_result = CurationFitResult(
        work_id="work",
        curves=curves,
        work_item=CurationWorkItem(
            work_id="work",
            dataset_name="Toy",
            group_key="group",
            chunk_index=None,
            input_table=pd.DataFrame(),
            config={},
            n_curves=1,
            input_filename="Toy_raw.csv",
        ),
    )

    dataset = combine([fit_result], dataset_name="Toy")

    assert "EC50_curvecurator" in dataset.columns
    assert "IC50_curvecurator" in dataset.columns
    assert dataset.loc["A|D", "cell_line_name"] == "A"

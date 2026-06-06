"""Split raw viability data into in-memory CurveCurator work items."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.curation._curvecurator.types import CurationSplitResult, CurationWorkItem


def load_raw_curve_df(input_path: Path | str) -> pd.DataFrame:
    """Load and validate a raw viability CSV.

    :param input_path: Path to raw viability data.
    :returns: Validated curve dataframe.
    :raises ValueError: If required columns are missing.
    """
    required_columns = {"dose", "response", "sample", "drug"}
    optional_columns = {"replicate"}
    allowed_columns = required_columns | optional_columns
    converters = {"dose": float, "response": float, "sample": str, "drug": str, "replicate": int}
    curve_df = pd.read_csv(
        input_path,
        usecols=lambda column: column in allowed_columns,
        converters=converters,
    )

    missing_columns = sorted(required_columns - set(curve_df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in viability data. Required columns are {sorted(required_columns)}.")
    return curve_df


def prepare_input_table(curve_df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[float], int, int]:
    """Transform raw curve rows into a CurveCurator input table.

    :param curve_df: Raw viability rows for one compatible group/chunk.
    :returns: Input table, experiment count, doses, replicate count, and curve count.
    """
    if "replicate" in curve_df.columns:
        n_replicates = curve_df["replicate"].nunique()
        pivot_columns = ["dose", "replicate"]
        duplicate_columns = ["sample", "drug", "dose", "replicate"]
    else:
        n_replicates = 1
        pivot_columns = ["dose"]
        duplicate_columns = ["sample", "drug", "dose"]

    working_df = curve_df
    if working_df.duplicated(subset=duplicate_columns).any():
        warnings.warn(
            "CurveCurator Raw Data Processing: Duplicate entries found for some sample/drug/dose"
            " combinations. Aggregating using mean of the 'response'.",
            UserWarning,
            stacklevel=2,
        )
        working_df = working_df.groupby(duplicate_columns, as_index=False)["response"].mean()

    df = working_df.pivot(index=["sample", "drug"], columns=pivot_columns, values="response")

    if "replicate" in working_df.columns:
        control_df = pd.DataFrame({(0.0, col_id): 1.0 for col_id in range(n_replicates)}, index=df.index)
    else:
        control_df = pd.DataFrame({0.0: 1.0}, index=df.index)

    df = pd.concat([control_df, df], axis=1)

    concentrations = df.columns.sort_values()
    doses = concentrations.get_level_values(0).to_list()
    df = df[concentrations]

    experiments = np.arange(df.shape[1])
    df.insert(0, "Name", ["|".join(map(str, index)) for index in df.index.tolist()])
    df.columns = ["Name"] + [f"Raw {i}" for i in experiments]

    return df, len(experiments), doses, n_replicates, len(df)


def build_config(
    filename: str,
    n_exp: int,
    n_replicates: int,
    doses: list[float],
    dataset_name: str,
    cores: int,
    n_curves: int,
    routing_device: str,
    condition: str = "",
    normalize: bool = False,
) -> dict:
    """Build a CurveCurator configuration dictionary.

    :param filename: Source raw input filename.
    :param n_exp: Number of experiments in the prepared input table.
    :param n_replicates: Number of replicates represented in the input table.
    :param doses: Dose values used by CurveCurator.
    :param dataset_name: Dataset name for metadata.
    :param cores: CPU worker count for CurveCurator processing config.
    :param routing_device: Device routing decision for this chunk.
    :param condition: Grouping condition label.
    :param normalize: Whether CurveCurator should normalize responses.
    :param n_curves: Curve count used for drevalpy device routing metadata.
    :returns: CurveCurator configuration dictionary.
    """
    config = {
        "Meta": {
            "id": filename,
            "description": dataset_name,
            "condition": condition,
            "treatment_time": "72 h",
        },
        "Experiment": {
            "experiments": list(range(n_exp)),
            "doses": doses,
            "dose_scale": "1e-06",
            "dose_unit": "uM",
            "control_experiment": [i for i in range(n_replicates)],
            "measurement_type": "OTHER",
            "data_type": "OTHER",
            "search_engine": "OTHER",
            "search_engine_version": "0",
        },
        "Paths": {
            "input_file": "curvecurator_input.tsv",
            "curves_file": "curves.tsv",
            "normalization_file": "norm.txt",
            "mad_file": "mad.txt",
            "dashboard": "dashboard.html",
        },
        "Processing": {
            "available_cores": cores,
            "max_missing": max(len(doses) - 5, 0),
            "imputation": False,
            "normalization": normalize,
        },
        "Curve Fit": {
            "type": "OLS",
            "speed": "exhaustive",
            "max_iterations": 1000,
            "interpolation": False,
            "control_fold_change": True,
        },
        "F Statistic": {
            "optimized_dofs": True,
            "alpha": 0.05,
            "fc_lim": 0.45,
        },
    }
    config["Routing"] = {"n_curves": n_curves, "device": routing_device}
    return config


def _iter_curve_groups(curve_df: pd.DataFrame):
    groupby: list[str] = []

    curve_df = curve_df.copy()
    curve_df["mindose"] = curve_df.groupby(["sample", "drug"], as_index=False)["dose"].transform("min")
    curve_df["maxdose"] = curve_df.groupby(["sample", "drug"], as_index=False)["dose"].transform("max")

    if curve_df["maxdose"].nunique() > 1:
        groupby.append("maxdose")
    if curve_df["mindose"].nunique() > 1:
        groupby.append("mindose")
    if "replicate" in curve_df.columns:
        curve_df["nreplicates"] = curve_df.groupby(["sample", "drug"])["replicate"].transform("nunique")
        if curve_df["nreplicates"].nunique() > 1:
            groupby.append("nreplicates")

    if groupby:
        yield from curve_df.groupby(groupby)
    else:
        yield ("drug_treatment", curve_df)


def _group_prefix(index) -> str:
    if isinstance(index, tuple):
        return "_".join(str(part) for part in index)
    return str(index)


def _split_group_into_chunks(
    curve_df: pd.DataFrame,
    *,
    effective_chunk: int,
) -> list[tuple[pd.DataFrame, int | None]]:
    """Split one group into in-memory chunks when larger than *effective_chunk*."""
    n_curves = curve_df[["sample", "drug"]].drop_duplicates().shape[0]
    if n_curves <= effective_chunk:
        return [(curve_df, None)]

    pairs = curve_df[["sample", "drug"]].drop_duplicates().sort_values(["sample", "drug"])
    chunks: list[tuple[pd.DataFrame, int | None]] = []
    n_chunks = math.ceil(n_curves / effective_chunk)
    for chunk_index in range(n_chunks):
        chunk_start = chunk_index * effective_chunk
        chunk_stop = (chunk_index + 1) * effective_chunk
        chunk_pairs = pairs.iloc[chunk_start:chunk_stop]
        chunk_df = curve_df.merge(chunk_pairs, on=["sample", "drug"], how="inner")
        chunks.append((chunk_df, chunk_index))
    return chunks


def _routing_device(
    requested: str,
    n_curves: int,
    gpu_min_curves: int,
    *,
    gpu_available: bool,
) -> str:
    """Decide whether a group/chunk is routed to CPU or accelerator-capable fitting."""
    if not gpu_available or requested == "cpu" or n_curves < gpu_min_curves:
        return "cpu"
    return requested


def _work_items_for_group(
    df: pd.DataFrame,
    *,
    group_key: str,
    input_filename: str,
    dataset_name: str,
    cores: int,
    normalize: bool,
    device: str,
    chunk_size: int,
    gpu_min_curves: int,
    gpu_chunk_size: int,
    gpu_available: bool,
    work_id_prefix: str,
) -> list[CurationWorkItem]:
    n_curves = df[["sample", "drug"]].drop_duplicates().shape[0]
    group_routing_device = _routing_device(device, n_curves, gpu_min_curves, gpu_available=gpu_available)
    effective_chunk = gpu_chunk_size if group_routing_device != "cpu" else chunk_size
    items: list[CurationWorkItem] = []

    for chunk_df, chunk_index in _split_group_into_chunks(df, effective_chunk=effective_chunk):
        chunk_n = chunk_df[["sample", "drug"]].drop_duplicates().shape[0]
        chunk_routing_device = _routing_device(device, chunk_n, gpu_min_curves, gpu_available=gpu_available)
        input_table, n_exp, doses, n_replicates, _ = prepare_input_table(chunk_df)
        condition = group_key if chunk_index is None else f"{group_key}_chunk_{chunk_index}"
        config = build_config(
            filename=input_filename,
            n_exp=n_exp,
            n_replicates=n_replicates,
            doses=doses,
            dataset_name=dataset_name,
            cores=min(chunk_n, cores),
            n_curves=chunk_n,
            routing_device=chunk_routing_device,
            condition=condition,
            normalize=normalize,
        )
        work_id = condition if chunk_index is None else f"{work_id_prefix}_{group_key}_chunk_{chunk_index}"
        items.append(
            CurationWorkItem(
                work_id=work_id,
                dataset_name=dataset_name,
                group_key=group_key,
                chunk_index=chunk_index,
                input_table=input_table,
                config=config,
                n_curves=chunk_n,
                input_filename=input_filename,
            )
        )
    return items


def split(
    raw_df: pd.DataFrame,
    *,
    dataset_name: str,
    input_filename: str,
    cores: int = 1,
    normalize: bool = False,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    gpu_available: bool = False,
) -> CurationSplitResult:
    """Split raw viability data into in-memory CurveCurator work items.

    :param raw_df: Raw viability table with dose, response, sample, and drug columns.
    :param dataset_name: Dataset name used in metadata and combine output.
    :param input_filename: Source filename recorded in work-item metadata.
    :param cores: Maximum CPU worker threads for CurveCurator processing config.
    :param normalize: Whether CurveCurator should normalize responses.
    :param device: Requested PyTorch device string for chunk sizing.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param gpu_available: Whether GPU resources are available for accelerator chunking.
    :returns: Prepared in-memory work items.
    """
    work_items: list[CurationWorkItem] = []

    for index, group_df in _iter_curve_groups(raw_df):
        group_key = _group_prefix(index)
        work_items.extend(
            _work_items_for_group(
                group_df,
                group_key=group_key,
                input_filename=input_filename,
                dataset_name=dataset_name,
                cores=cores,
                normalize=normalize,
                device=device,
                chunk_size=chunk_size,
                gpu_min_curves=gpu_min_curves,
                gpu_chunk_size=gpu_chunk_size,
                gpu_available=gpu_available,
                work_id_prefix=dataset_name,
            )
        )

    return CurationSplitResult(
        dataset_name=dataset_name,
        input_filename=input_filename,
        work_items=tuple(work_items),
    )

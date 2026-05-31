"""
Contains all function required for CurveCurator fitting.

CurveCurator publication:
Bayer, F.P., Gander, M., Kuster, B. et al. CurveCurator: a recalibrated F-statistic to assess,
classify, and explore significance of dose–response curves. Nat Commun 14, 7902 (2023).
https://doi-org.eaccess.tum.edu/10.1038/s41467-023-43696-z

CurveCurator applies a recalibrated F-statistic for p-value estimation of 4-point log-logistic
regression fits. In drevalpy, this can be used to generate training data with higher quality, since
quality measures, such as p-value, R2, or relevance score can be used to filter out viability
measurements of low quality.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.datasets.curvecurator_device import effective_device
from drevalpy.datasets.curvecurator_runner import (
    CurveCuratorWorkItem,
    run_curvecurator_work_items,
    split_group_into_chunks,
)
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER

from ..pipeline_function import pipeline_function


def _prepare_raw_data(curve_df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    if "replicate" in curve_df.columns:
        # Replicates are pooled into one CurveCurator row per (sample, drug);
        # they become additional Raw columns, not separate per-replicate fits.
        n_replicates = curve_df["replicate"].nunique()
        pivot_columns = ["dose", "replicate"]
        duplicate_columns = ["sample", "drug", "dose", "replicate"]
    else:
        n_replicates = 1
        pivot_columns = ["dose"]
        duplicate_columns = ["sample", "drug", "dose"]

    if curve_df.duplicated(subset=duplicate_columns).any():
        warnings.warn(
            "CurveCurator Raw Data Processing: Duplicate entries found for some sample/drug/dose"
            " combinations. Aggregating using mean of the 'response'.",
            UserWarning,
            stacklevel=1,
        )
        curve_df = curve_df.groupby(duplicate_columns, as_index=False)["response"].mean()

    df = curve_df.pivot(index=["sample", "drug"], columns=pivot_columns, values="response")

    if "replicate" in curve_df.columns:
        control_df = pd.DataFrame({(0.0, col_id): 1.0 for col_id in range(n_replicates)}, index=df.index)
    else:
        control_df = pd.DataFrame({0.0: 1.0}, index=df.index)

    df = pd.concat([control_df, df], axis=1)

    concentrations = df.columns.sort_values()
    doses = concentrations.get_level_values(0).to_list()
    df = df[concentrations]

    experiments = np.arange(df.shape[1])
    df.insert(0, "Name", ["|".join(map(str, i)) for i in df.index.tolist()])

    df.columns = ["Name"] + [f"Raw {i}" for i in experiments]

    curvecurator_folder = output_dir / prefix
    curvecurator_folder.mkdir(exist_ok=True, parents=True)
    df.to_csv(curvecurator_folder / "curvecurator_input.tsv", sep="\t", index=False)

    return len(experiments), doses, n_replicates, len(df)


def _build_config(
    filename: str,
    n_exp: int,
    n_replicates: int,
    doses: list[float],
    dataset_name: str,
    cores: int,
    condition: str = "",
    normalize: bool = False,
    n_curves: int | None = None,
) -> dict:
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
    if n_curves is not None:
        config["Routing"] = {"n_curves": n_curves}
    return config


def _load_raw_curve_df(input_path: Path) -> pd.DataFrame:
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


def _work_items_for_group(
    df: pd.DataFrame,
    output_path: Path,
    prefix: str,
    *,
    input_filename: str,
    dataset_name: str,
    cores: int,
    normalize: bool,
    device: str,
    chunk_size: int,
    gpu_min_curves: int,
    gpu_chunk_size: int,
) -> list[CurveCuratorWorkItem]:
    n_curves = df[["sample", "drug"]].drop_duplicates().shape[0]
    eff_device = effective_device(device, n_curves, gpu_min_curves)
    effective_chunk = gpu_chunk_size if eff_device != "cpu" else chunk_size
    group_dir = output_path / prefix
    items: list[CurveCuratorWorkItem] = []

    for chunk_df, chunk_dir in split_group_into_chunks(
        df,
        group_dir,
        effective_chunk=effective_chunk,
    ):
        chunk_n = chunk_df[["sample", "drug"]].drop_duplicates().shape[0]
        n_exp, doses, n_replicates, _ = _prepare_raw_data(
            curve_df=chunk_df,
            output_dir=chunk_dir.parent,
            prefix=chunk_dir.name if chunk_dir != group_dir else prefix,
        )
        config = _build_config(
            filename=input_filename,
            n_exp=n_exp,
            n_replicates=n_replicates,
            doses=doses,
            dataset_name=dataset_name,
            cores=min(chunk_n, cores),
            condition=prefix,
            normalize=normalize,
            n_curves=chunk_n,
        )
        items.append(CurveCuratorWorkItem(chunk_dir=chunk_dir, config=config, n_curves=chunk_n))
    return items


def _prepare_work_items(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    cores: int,
    *,
    normalize: bool = False,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
) -> list[CurveCuratorWorkItem]:
    curve_df = _load_raw_curve_df(input_path)
    work_items: list[CurveCuratorWorkItem] = []

    for index, df in _iter_curve_groups(curve_df):
        prefix = _group_prefix(index)
        work_items.extend(
            _work_items_for_group(
                df,
                output_path,
                prefix,
                input_filename=input_path.name,
                dataset_name=dataset_name,
                cores=cores,
                normalize=normalize,
                device=device,
                chunk_size=chunk_size,
                gpu_min_curves=gpu_min_curves,
                gpu_chunk_size=gpu_chunk_size,
            )
        )
    return work_items


def _calc_ic50(model_params_df: pd.DataFrame):
    """
    Calculate the IC50 in M from a fitted model.

    This function expects a dataframe that was processed in the postprocess function, containing
    the columns "Front", "Back", "Slope", "pEC50". It calculates the IC50 for all the models in the
    dataframe in closed form and adds the column IC50_curvecurator to the input dataframe.
    Also adds the natural logarithm of the IC50 as LN_IC50_curvecurator.

    :param model_params_df: a dataframe containing the fitted parameters
    """

    def ic50(front, back, slope, pec50):
        with np.errstate(invalid="ignore"):
            return np.power(10, (np.log10((front - 0.5) / (0.5 - back)) - slope * pec50) / slope)

    front = model_params_df["Front"].values
    back = model_params_df["Back"].values
    slope = model_params_df["Slope"].values
    # we need the pEC50 in uM; now it is in M: -log10(EC50[M] * 10^6) = -log10(EC50[M])-6 = pEC50 -6
    pec50 = model_params_df["pEC50_curvecurator"].values - 6

    model_params_df["IC50_curvecurator"] = ic50(front, back, slope, pec50)
    model_params_df["LN_IC50_curvecurator"] = np.log(model_params_df["IC50_curvecurator"].values)


@pipeline_function
def postprocess(output_folder: str, dataset_name: str):
    """
    Postprocess CurveCurator output files.

    This function reads all curves.tsv files created by CurveCurator, which contain the
    fitted curve parameters, postprocesses them to be used by drevalpy and combines everything
    in one <dataset_name>.csv file for usage by drevalpy.

    :param output_folder: Path to the output folder of CurveCurator containing the curves.txt file.
    :param dataset_name: The name of the dataset, will be used to prepend the postprocessed <dataset_name>.csv file
    """
    output_path = Path(output_folder)
    curvecurator_output_files = output_path.rglob("curves.tsv")
    required_columns = {
        "Name": "Name",
        "pEC50": "pEC50_curvecurator",
        "pEC50 Error": "pEC50Error",
        "Curve Slope": "Slope",
        "Curve Front": "Front",
        "Curve Back": "Back",
        "Curve Fold Change": "FoldChange",
        "Curve AUC": "AUC_curvecurator",
        "Curve R2": "R2",
        "Curve P_Value": "pValue",
        "Curve Relevance Score": "RelevanceScore",
        "Curve F_Value": "fValue",
        "Curve Log P_Value": "negLog10pValue",
        "Signal Quality": "SignalQuality",
        "Curve RMSE": "RMSE",
        "Curve F_Value SAM Corrected": "fValueSAMCorrected",
        "Curve Regulation": "Regulation",
    }

    with open(output_path / f"{dataset_name}.csv", "w") as f:
        first_file = True
        for output_file in curvecurator_output_files:
            fitted_curve_data = pd.read_csv(output_file, sep="\t", usecols=required_columns).rename(
                columns=required_columns
            )
            fitted_curve_data[[CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER]] = fitted_curve_data.Name.str.split(
                "|", expand=True
            )
            fitted_curve_data["EC50_curvecurator"] = (
                np.power(10, -fitted_curve_data["pEC50_curvecurator"].values) * 10**6
            )  # in CurveCurator 10^-pEC50 = EC50
            _calc_ic50(fitted_curve_data)
            fitted_curve_data.to_csv(f, index=None, header=first_file, mode="a")
            first_file = False
        f.close()


def fit_curves(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    cores: int,
    normalize: bool = False,
    *,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
):
    """
    Fit curves for provided raw viability data.

    This functions reads viability data in a predefined input format, preprocesses the data
    to be readable by CurveCurator, fits curves using the fork's Python API, and postprocesses
    the fitted data to a format required by drevalpy.

    :param input_file: Path to the file containing the raw viability data
    :param output_dir: Path to store CurveCurator input, output files, and postprocessed data.
    :param dataset_name: The name of the dataset, will be used to prepend the postprocessed <dataset_name>.csv file
    :param cores: The number of cores to use for CPU chunk concurrency during fitting.
    :param normalize: Whether to normalize the response values to [0, 1] for curvecurator. Default = False.
    :param device: PyTorch device for fitting: ``auto``, ``cpu``, ``cuda``, ``cuda:0``, or ``mps``.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    work_items = _prepare_work_items(
        input_path,
        output_path,
        dataset_name,
        cores,
        normalize=normalize,
        device=device,
        chunk_size=chunk_size,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
    )
    run_curvecurator_work_items(
        work_items,
        cores=cores,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
    )
    postprocess(output_folder=output_dir, dataset_name=dataset_name)

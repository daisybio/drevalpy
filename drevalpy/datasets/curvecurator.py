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

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from ..pipeline_function import pipeline_function


def _prepare_raw_data(curve_df: pd.DataFrame, output_dir: str | Path):
    required_columns = ["dose", "response", "sample", "drug"]
    if not all([col in curve_df.columns for col in required_columns]):
        raise ValueError("Missing columns in viability data. Required columns are {required_columns}.")
    if "replicate" in curve_df.columns:
        required_columns.append("replicate")
    curve_df = curve_df[required_columns]
    n_replicates = 1
    conc_columns = ["dose"]
    has_multicol_index = False
    if "replicate" in curve_df.columns:
        n_replicates = curve_df["replicate"].nunique()
        conc_columns.append("replicate")
        has_multicol_index = True

    df = curve_df.pivot(index=["sample", "drug"], columns=conc_columns, values="response")

    for i in range(n_replicates):
        df.insert(0, (0.0, n_replicates - i), 1.0)

    concentrations = df.columns.sort_values()
    df = df[concentrations]

    experiments = np.arange(df.shape[1])
    df.insert(0, "Name", df.index.map(lambda x: f"{x[0]}|{x[1]}"))
    df.columns = ["Name"] + [f"Raw {i}" for i in experiments]

    curvecurator_folder = Path(output_dir)
    curvecurator_folder.mkdir(exist_ok=True, parents=True)
    df.to_csv(curvecurator_folder / "curvecurator_input.tsv", sep="\t", index=False)

    if has_multicol_index:
        doses = [pair[0] for pair in concentrations]
    else:
        doses = concentrations.to_list()
    return len(experiments), doses, n_replicates, len(df)


def _prepare_toml(filename: str, n_exp: int, n_replicates: int, doses: list[float], dataset_name: str, cores: int):
    config = {
        "Meta": {
            "id": filename,
            "description": dataset_name,
            "condition": "drug",
            "treatment_time": "72 h",
        },
        "Experiment": {
            "experiments": range(n_exp),
            "doses": doses,
            "dose_scale": "1e-06",
            "dose_unit": "M",
            "control_experiment": [i for i in range(n_replicates)],
            "measurement_type": "OTHER",
            "data_type": "OTHER",
            "search_engine": "OTHER",
            "search_engine_version": "0",
        },
        "Paths": {
            "input_file": "curvecurator_input.tsv",
            "curves_file": "curves.txt",
            "normalization_file": "norm.txt",
            "mad_file": "mad.txt",
            "dashboard": "dashboard.html",
        },
        "Processing": {
            "available_cores": cores,
            "max_missing": max(len(doses) - 5, 0),
            "imputation": False,
            "normalization": False,
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
    return config


def _exec_curvecurator(output_dir: Path):
    command = ["CurveCurator", str(output_dir / "config.toml"), "--mad"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()


def _calc_ic50(model_params_df: pd.DataFrame):
    """
    Calculate the IC50 from a fitted model.

    This function expects a dataframe that was processed in the postprocess function, containing
    the columns "Front", "Back", "Slope", "pEC50". It calculates the IC50 for all the models in the
    dataframe in closed form and adds the column IC50_curvecurator to the input dataframe.

    :param model_params_df: a dataframe containing the fitted parameters
    """

    def ic50(front, back, slope, pec50):
        return (np.log10((front - back) / (0.5 + back)) - slope * pec50) / slope

    front = model_params_df["Front"].values
    back = model_params_df["Back"].values
    slope = model_params_df["Slope"].values
    pec50 = model_params_df["pEC50_curvecurator"].values

    model_params_df["IC50_curvecurator"] = ic50(front, back, slope, pec50)


@pipeline_function
def preprocess(input_file: str | Path, output_dir: str | Path, dataset_name: str, cores: int):
    """
    Preprocess raw viability data and create config.toml for use with CurveCurator.

    :param input_file: Path to csv file containing the raw viability data
    :param output_dir: Path to store all the files to, including the preprocessed data, the config.toml
        for CurveCurator, CurveCurator's output files, and the postprocessed data
    :param dataset_name: Name of the dataset
    :param cores: The number of cores to be used for fitting the curves using CurveCurator.
        This parameter is written into the config.toml, but it is min of the number of curves to fit
        and the number given (min(n_curves, cores))
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    curve_df = pd.read_csv(input_file)

    n_exp, doses, n_replicates, n_curves_to_fit = _prepare_raw_data(curve_df, output_dir)
    cores = min(n_curves_to_fit, cores)

    config = _prepare_toml(input_file.name, n_exp, n_replicates, doses, dataset_name, cores)
    with open(output_dir / "config.toml", "w") as f:
        toml.dump(config, f)


@pipeline_function
def postprocess(output_folder: str | Path, dataset_name: str):
    """
    Postprocess CurveCurator output file.

    This function reads the curves.txt file created by CurveCurator, which contains the
    fitted curve parameters and postprocesses it to be used by drevalpy.

    :param output_folder: Path to the output folder of CurveCurator containing the curves.txt file.
    :param dataset_name: The name of the dataset, will be used to prepend the postprocessed <dataset_name>.csv file
    """
    output_folder = Path(output_folder)
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
    fitted_curve_data = pd.read_csv(Path(output_folder) / "curves.txt", sep="\t", usecols=required_columns).rename(
        columns=required_columns
    )
    fitted_curve_data[["cell_line_id", "drug_id"]] = fitted_curve_data.Name.str.split("|", expand=True)
    fitted_curve_data["EC50_curvecurator"] = np.power(
        10, -fitted_curve_data["pEC50_curvecurator"].values
    )  # in CurveCurator 10^-pEC50 = EC50
    _calc_ic50(fitted_curve_data)
    fitted_curve_data.to_csv(output_folder / f"{dataset_name}.csv", index=None)


def fit_curves(input_file: str | Path, output_dir: str | Path, dataset_name: str, cores: int):
    """
    Fit curves for provided raw viability data.

    This functions reads viability data in a predefined input format, preprocesses the data
    to be readable by CurveCurator, fits curves to the data using CurveCurator, and postprocesses
    the fitted data to a format required by drevalpy.

    :param input_file: Path to the file containing the raw viability data
    :param output_dir: Path to store all the files to, including the preprocessed data, the config.toml
        for CurveCurator, CurveCurator's output files, and the postprocessed data
    :param dataset_name: The name of the dataset, will be used to prepend the postprocessed <dataset_name>.csv file
    :param cores: The number of cores to be used for fitting the curves using CurveCurator.
        This parameter is written into the config.toml, but it is min of the number of curves to fit
        and the number given (min(n_curves, cores))
    """
    preprocess(input_file, output_dir, dataset_name, cores)
    _exec_curvecurator(Path(output_dir))
    postprocess(output_dir, dataset_name)

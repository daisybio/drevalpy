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


def _prepare_raw_data(curve_df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    if "replicate" in curve_df.columns:
        n_replicates = curve_df["replicate"].nunique()
        pivot_columns = ["dose", "replicate"]
    else:
        n_replicates = 1
        pivot_columns = ["dose"]

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
    df.reset_index(drop=True)

    df.columns = ["Name"] + [f"Raw {i}" for i in experiments]

    curvecurator_folder = output_dir / prefix
    curvecurator_folder.mkdir(exist_ok=True, parents=True)
    df.to_csv(curvecurator_folder / "curvecurator_input.tsv", sep="\t", index=False)

    return len(experiments), doses, n_replicates, len(df)


def _prepare_toml(
    filename: str, n_exp: int, n_replicates: int, doses: list[float], dataset_name: str, cores: int, condition: str = ""
):
    config = {
        "Meta": {
            "id": filename,
            "description": dataset_name,
            "condition": condition,
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
            "curves_file": "curves.tsv",
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


def _exec_curvecurator(output_dir: Path, batched: bool = True):
    """
    Execute CurveCurator in batch mode.

    This function spawns a subprocess that runs CurveCurator for all config.toml files that
    are listed in a file "configlist.txt" in the provided output directory.

    :param output_dir: The directory containing einter configlist.txt as well as subfolders for
        all the paths listed in configlist.txt that function as input and output directories for
        batched CurveCurator execution, or the directory containig a single config.toml and
        corresponding viability input.
    :param batched: If True, run CurveCurator in batched mode (default), iterating over a list
        of configs spefified in <output_dir>/configlist.txt and consecutively executing each
        CurveCurator run. If False, run a single CurveCurator run (this can be used for
        parallelisation).
    """
    if batched:
        command = ["CurveCurator", str(output_dir / "configlist.txt"), "--mad", "--batch"]
    else:
        command = ["CurveCurator", str(output_dir / "config.toml"), "--mad"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()


def _calc_ic50(model_params_df: pd.DataFrame):
    """
    Calculate the IC50 in M from a fitted model.

    This function expects a dataframe that was processed in the postprocess function, containing
    the columns "Front", "Back", "Slope", "pEC50". It calculates the IC50 for all the models in the
    dataframe in closed form and adds the column IC50_curvecurator to the input dataframe.

    :param model_params_df: a dataframe containing the fitted parameters
    """

    def ic50(front, back, slope, pec50):
        with np.errstate(invalid="ignore"):
            return np.power(10, (np.log10((front - 0.5) / (0.5 - back)) - slope * pec50) / slope)

    front = model_params_df["Front"].values
    back = model_params_df["Back"].values
    slope = model_params_df["Slope"].values
    pec50 = model_params_df["pEC50_curvecurator"].values

    model_params_df["IC50_curvecurator"] = ic50(front, back, slope, pec50)


@pipeline_function
def preprocess(input_file: str | Path, output_dir: str | Path, dataset_name: str, cores: int):
    """
    Preprocess raw viability data and create required input files for CurveCurator.

    This function takes an input file containing raw viability in long format. The required columns
    are "dose", "response", "sample", and "drug", with an optional "replicate" column.
    If there are multiple dose ranges or numbers of replicates, groups in the form
    (maxdose, mindose, n_replicates) are created to keep the number of parameters for fitting low
    and the input dataframes for curvecurator as dense as possible.
    All dosages must be provided in µM!
    All responses must be normalized against the control already without the response for the control.

    :param input_file: Path to csv file containing the raw viability data
    :param output_dir: Path to store all the files to, including the preprocessed data, the config.toml
        for CurveCurator, CurveCurator's output files, and the postprocessed data
    :param dataset_name: Name of the dataset
    :param cores: The number of cores to be used for fitting the curves using CurveCurator.
        This parameter is written into the config.toml, but it is min of the number of curves to fit
        and the number given (min(n_curves, cores))
    :raises ValueError: If required columns are not found in the provided input file.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    required_columns = ["dose", "response", "sample", "drug", "replicate"]
    converters = {"dose": float, "response": float, "sample": str, "drug": str, "replicate": int}
    try:
        curve_df = pd.read_csv(input_file, usecols=required_columns, converters=converters)
    except ValueError:
        required_columns.pop()
        del converters["replicate"]
        curve_df = pd.read_csv(input_file, usecols=required_columns, converters=converters)

    if not all([col in curve_df.columns for col in required_columns]):
        raise ValueError(f"Missing columns in viability data. Required columns are {required_columns}.")
    groupby = []

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

    if len(groupby) > 0:
        drug_df_groups = curve_df.groupby(groupby)
    else:
        drug_df_groups = [("drug_treatment", curve_df)]

    configs = []

    for index, df in drug_df_groups:
        prefix = "_".join([f"{s}" for s in index])
        n_exp, doses, n_replicates, n_curves_to_fit = _prepare_raw_data(
            curve_df=df, output_dir=output_dir, prefix=prefix
        )
        config = _prepare_toml(
            input_file.name, n_exp, n_replicates, doses, dataset_name, min(n_curves_to_fit, cores), prefix
        )
        config_path = output_dir / prefix / "config.toml"
        with open(config_path, "w") as f:
            toml.dump(config, f)
        configs.append(f"{config_path}\n")

    with open(output_dir / "configlist.txt", "w") as f:
        f.writelines(configs)


@pipeline_function
def postprocess(output_folder: str | Path, dataset_name: str):
    """
    Postprocess CurveCurator output files.

    This function reads all curves.tsv files created by CurveCurator, which contain the
    fitted curve parameters, postprocesses them to be used by drevalpy and combines everything
    in one <dataset_name>.csv file for usage by drevalpy.

    :param output_folder: Path to the output folder of CurveCurator containing the curves.txt file.
    :param dataset_name: The name of the dataset, will be used to prepend the postprocessed <dataset_name>.csv file
    """
    output_folder = Path(output_folder)
    curvecurator_output_files = output_folder.rglob("curves.tsv")
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

    with open(output_folder / f"{dataset_name}.csv", "w") as f:
        first_file = True
        for output_file in curvecurator_output_files:
            fitted_curve_data = pd.read_csv(output_file, sep="\t", usecols=required_columns).rename(
                columns=required_columns
            )
            fitted_curve_data[["cell_line_id", "drug_id"]] = fitted_curve_data.Name.str.split("|", expand=True)
            fitted_curve_data["EC50_curvecurator"] = np.power(
                10, -fitted_curve_data["pEC50_curvecurator"].values
            )  # in CurveCurator 10^-pEC50 = EC50
            _calc_ic50(fitted_curve_data)
            fitted_curve_data.to_csv(f, index=None, header=first_file, mode="a")
            first_file = False
        f.close()


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

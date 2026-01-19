"""For the nf-core/drugresponseeval subworkflow run_cv."""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import yaml


def load_response():
    """CLI for loading the drug response data."""
    from drevalpy.datasets.dataset import DrugResponseDataset
    from drevalpy.datasets.loader import AVAILABLE_DATASETS
    from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER

    # define parser
    parser = argparse.ArgumentParser(description="Load data for drug response prediction as pickle.")
    parser.add_argument(
        "--response_dataset", type=str, required=True, help="Path to the drug response file dataset_name.csv."
    )
    parser.add_argument(
        "--cross_study_dataset",
        action="store_true",
        default=False,
        help="Whether to load cross-study datasets, default: False.",
    )
    parser.add_argument(
        "--measure",
        type=str,
        default="LN_IC50_curvecurator",
        help="Name of the column in the dataset containing the drug response measures, default: LN_IC50_curvecurator.",
    )
    args = parser.parse_args()

    # read in the csv and create the DrugResponseDataset
    dataset_name = Path(args.response_dataset).stem
    input_file = Path(f"{dataset_name}.csv")
    if dataset_name in AVAILABLE_DATASETS:
        response_file = pd.read_csv(input_file, dtype={"pubchem_id": str})
        if dataset_name == "BeatAML2":
            # only has AML patients = blood
            response_file[TISSUE_IDENTIFIER] = "Blood"
        elif dataset_name == "PDX_Bruna":
            # only has breast cancer patients
            response_file[TISSUE_IDENTIFIER] = "Breast"
        response_data = DrugResponseDataset(
            response=response_file[args.measure].values,
            cell_line_ids=response_file[CELL_LINE_IDENTIFIER].values,
            drug_ids=response_file[DRUG_IDENTIFIER].values,
            tissues=response_file[TISSUE_IDENTIFIER].values,
            dataset_name=dataset_name,
        )
    else:
        tissue_column = TISSUE_IDENTIFIER
        # check whether the input file has a TISSUE_IDENTIFIER column, if not, set tissue_column to None
        if TISSUE_IDENTIFIER not in pd.read_csv(input_file, nrows=1).columns:
            tissue_column = None

        response_data = DrugResponseDataset.from_csv(
            input_file=input_file, dataset_name=dataset_name, measure=args.measure, tissue_column=tissue_column
        )
    outfile = f"cross_study_{dataset_name}.pkl" if args.cross_study_dataset else "response_dataset.pkl"
    # Pickle the object to a file
    with open(outfile, "wb") as f:
        pickle.dump(response_data, f)


def cv_split():
    """CLI for splitting the response.pkl into CV splits."""
    # define the parser
    parser = argparse.ArgumentParser(description="Split data into CV splits: split_0.pkl, split_1.pkl, ...")
    parser.add_argument("--response", type=str, required=True, help="Path to the pickled response data file.")
    parser.add_argument("--n_cv_splits", type=int, required=True, help="Number of CV splits")
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LTO, LDO), default: LPO.")
    parser.add_argument("--validation_ratio", type=float, default=0.1, help="Ratio of validation data, default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting the data, default: 42.")
    args = parser.parse_args()

    # load the response data and split it into CV splits
    with open(args.response, "rb") as f:
        response_data = pickle.load(f)
    response_data.remove_nan_responses()
    response_data.split_dataset(
        n_cv_splits=args.n_cv_splits,
        mode=args.test_mode,
        split_validation=True,
        split_early_stopping=True,
        validation_ratio=args.validation_ratio,
        random_state=args.seed,
    )

    # save the CV splits as pickled files
    for split_index, split in enumerate(response_data.cv_splits):
        with open(f"split_{split_index}.pkl", "wb") as f:
            pickle.dump(split, f)


def hpam_split():
    """CLI for creating hyperparameter yamls for the specified model.

    :raises ValueError: if the model_name is neither in the MULTI- nor in the SINGLE_DRUG_MODEL_FACTORY
    """
    from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY

    # define the parser
    parser = argparse.ArgumentParser(
        description="Takes the model name and creates one yaml for each unique hyperparameter combination "
        "(hpam_0.yaml, hpam_1.yaml, ...)."
    )
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument(
        "--hyperparameter_tuning",
        action="store_true",
        default=False,
        help="If set, hyperparameter tuning is performed, otherwise only the first combination is used",
    )
    args = parser.parse_args()

    # load the relevant parameters and instantiate the model
    if args.model_name in MULTI_DRUG_MODEL_FACTORY:
        model_name = args.model_name
    else:
        model_name = str(args.model_name).split(".")[0]
        if model_name not in SINGLE_DRUG_MODEL_FACTORY:
            raise ValueError(
                f"{model_name} neither in " f"SINGLE_DRUG_MODEL_FACTORY nor in " f"MULTI_DRUG_MODEL_FACTORY."
            )
    model_class = MODEL_FACTORY[model_name]
    # get all hyperparameter combinations
    hyperparameters = model_class.get_hyperparameter_set()
    if not args.hyperparameter_tuning:
        hyperparameters = [hyperparameters[0]]
    # save the hyperparameter combinations as yaml files
    hpam_idx = 0
    for hpam_combi in hyperparameters:
        with open(f"hpam_{hpam_idx}.yaml", "w") as yaml_file:
            hpam_idx += 1
            yaml.dump(hpam_combi, yaml_file, default_flow_style=False)


def train_and_predict_cv():
    """CLI for training and predicting on CV splits."""
    from drevalpy.experiment import get_datasets_from_cv_split, get_model_name_and_drug_id, train_and_predict
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    # define parser
    parser = argparse.ArgumentParser(
        description="Trains the specified model on the specified CV split with the specified hyperparameter "
        "configuration. Saves the prediction into a pickled file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
    )
    parser.add_argument("--path_data", type=str, default="data", help="Data directory path, default: data.")
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LTO, LDO), default: LPO.")
    parser.add_argument(
        "--hyperparameters",
        type=str,
        help="Path to the yaml file containing the hyperparameter configuration for this run.",
    )
    parser.add_argument("--cv_data", type=str, help="Path to the pickled cv data split.")
    parser.add_argument(
        "--response_transformation",
        type=str,
        default="None",
        help="Response transformation to apply to the dataset, default: None.",
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="TEMPORARY",
        help="model checkpoint directory, if not provided: temporary directory is used",
    )
    args = parser.parse_args()

    # load the relevant parameters
    model_name, drug_id = get_model_name_and_drug_id(args.model_name)

    model_class = MODEL_FACTORY[model_name]
    with open(args.cv_data, "rb") as f:
        split = pickle.load(f)

    train_dataset, validation_dataset, es_dataset, test_dataset = get_datasets_from_cv_split(
        split, model_class, model_name, drug_id
    )

    response_transform = get_response_transformation(args.response_transformation)
    with open(args.hyperparameters) as f:
        hpams = yaml.safe_load(f)
    model = model_class()

    # train and predict on validation dataset
    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        path_data=args.path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=es_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=args.model_checkpoint_dir,
    )

    # save the predictions
    with open(
        f"prediction_dataset_{model_name}_{str(args.cv_data).split('.pkl')[0]}_"
        f"{str(args.hyperparameters).split('.yaml')[0]}.pkl",
        "wb",
    ) as f:
        pickle.dump(validation_dataset, f)


def _best_metric(metric, current_metric, best_metric, minimization_metrics, maximization_metrics):
    # returns whether the current metric is better than the best metric based on the metric type.
    if metric in minimization_metrics:
        if current_metric < best_metric:
            return True
    elif metric in maximization_metrics:
        if current_metric > best_metric:
            return True
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return False


def evaluate_and_find_max():
    """CLI to evaluate the predictions and find the best hyperparameter combination."""
    from drevalpy.evaluation import MAXIMIZATION_METRICS, MINIMIZATION_METRICS, evaluate

    # define parser
    parser = argparse.ArgumentParser(
        description="Evaluates the predictions of the specified model on the specified CV split over all "
        "hyperparameter configurations. Identifies the best hyperparameter combination based "
        "on the specified metric and saves it into a yaml file."
    )
    parser.add_argument("--model_name", type=str, help="Model name, used for naming the output file.")
    parser.add_argument("--split_id", type=str, help="Split id, used for naming the output file.")
    parser.add_argument("--hpam_yamls", nargs="+", help="List of paths to hyperparameter configuration yaml files.")
    parser.add_argument("--pred_datas", nargs="+", help="List of paths to pickled predictions.")
    parser.add_argument("--optim_metric", type=str, default="RMSE", help="Optimization metric, default: RMSE.")
    args = parser.parse_args()

    # prepare data
    hpam_yamls = []
    for hpam_yaml in args.hpam_yamls:
        hpam_yamls.append(hpam_yaml)
    pred_datas = []
    for pred_data in args.pred_datas:
        pred_datas.append(pred_data)

    # find best hpam combi
    best_hpam_combi = None
    best_result = None
    for i in range(0, len(pred_datas)):
        with open(pred_datas[i], "rb") as pred_file:
            pred_data = pickle.load(pred_file)
        with open(hpam_yamls[i]) as yaml_file:
            hpam_combi = yaml.safe_load(yaml_file)
        results = evaluate(pred_data, args.optim_metric)
        if best_result is None:
            best_result = results[args.optim_metric]
            best_hpam_combi = hpam_combi
        elif _best_metric(
            metric=args.optim_metric,
            current_metric=results[args.optim_metric],
            best_metric=best_result,
            minimization_metrics=MINIMIZATION_METRICS,
            maximization_metrics=MAXIMIZATION_METRICS,
        ):
            best_result = results[args.optim_metric]
            best_hpam_combi = hpam_combi
    final_result = {
        f"{args.model_name}_{args.split_id}": {"best_hpam_combi": best_hpam_combi, "best_result": best_result}
    }
    with open(f"best_hpam_combi_{args.split_id}.yaml", "w") as yaml_file:
        yaml.dump(final_result, yaml_file, default_flow_style=False)

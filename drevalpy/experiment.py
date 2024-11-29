"""Main module for running the drug response prediction experiment."""

import json
import os
import shutil
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune
from sklearn.base import TransformerMixin

from .datasets.dataset import DrugResponseDataset, FeatureDataset
from .evaluation import evaluate, get_mode
from .models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY
from .models.drp_model import DRPModel
from .pipeline_function import pipeline_function


def drug_response_experiment(
    models: list[type[DRPModel]],
    response_data: DrugResponseDataset,
    baselines: Optional[list[type[DRPModel]]] = None,
    response_transformation: Optional[TransformerMixin] = None,
    run_id: str = "",
    test_mode: str = "LPO",
    metric: str = "RMSE",
    n_cv_splits: int = 5,
    multiprocessing: bool = False,
    randomization_mode: Optional[list[str]] = None,
    randomization_type: str = "permutation",
    cross_study_datasets: Optional[list[DrugResponseDataset]] = None,
    n_trials_robustness: int = 0,
    path_out: str = "results/",
    overwrite: bool = False,
    path_data: str = "data",
) -> None:
    """
    Run the drug response prediction experiment. Save results to disc.

    :param models: list of model classes to compare
    :param baselines: list of baseline models. No randomization or robustness tests are run for the baseline models.
    :param response_data: drug response dataset
    :param response_transformation: normalizer to use for the response data
    :param metric: metric to use for hyperparameter optimization
    :param n_cv_splits: number of cross-validation splits
    :param multiprocessing: whether to use multiprocessing
    :param randomization_mode: list of randomization modes to do. Modes: SVCC, SVRC, SVCD, SVRD Can be a list of
        randomization tests e.g. 'SVCC SVCD'. Default is None, which means no randomization tests are run.

        * SVCC: Single View Constant for Cell Lines: in this mode, one experiment is done for every cell line view
            the model uses (e.g. gene expression, mutation, ...). For each experiment one cell line view is held
            constant while the others are randomized.
        * SVRC Single View Random for Cell Lines: in this mode, one experiment is done for every cell line view the
            model uses (e.g. gene expression, mutation, ...). For each experiment one cell line view is randomized while
            the others are held constant.
        * SVCD: Single View Constant for Drugs: in this mode, one experiment is done for every drug view the model
            uses (e.g. fingerprints, target_information, ...). For each experiment one drug view is held constant
            while the others are randomized.
        * SVRD: Single View Random for Drugs: in this mode, one experiment is done for every drug view the model uses
            (e.g. gene expression, target_information, ...). For each experiment one drug view is randomized while
            the others are held constant.

    :param randomization_type: type of randomization to use. Choose from "permutation" and "invariant".
        Default is "permutation".

        * "permutation": permute the features over the instances, keeping the distribution of the features the same
            but dissolving the relationship to the target
        * "invariant": the features are permuted in a way that a key characteristic of the feature is kept. In case of
            matrices, this is the mean and standard deviation of the feature view for this instance, for networks it
            is the degree distribution.

    :param cross_study_datasets: list of datasets for the cross-study prediction. The trained model is assessed for
        its generalization to these datasets. Default is None, which means no cross-study prediction is run.
    :param n_trials_robustness: number of trials to run for the robustness test. The robustness test is a test where
        models are retrained multiple times with varying seeds. Default is 0, which means no robustness test is run.
    :param path_out: path to the output directory
    :param run_id: identifier to save the results
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param overwrite: whether to overwrite existing results
    :param path_data: path to the data directory, usually data/
    :raises ValueError: if no cv splits are found
    """
    if baselines is None:
        baselines = []
    cross_study_datasets = cross_study_datasets or []
    result_path = os.path.join(path_out, run_id, test_mode)
    split_path = os.path.join(result_path, "splits")
    result_folder_exists = os.path.exists(result_path)
    if result_folder_exists and overwrite:
        # if results exists, delete them if overwrite is True
        print(f"Overwriting existing results at {result_path}")
        shutil.rmtree(result_path)

    if result_folder_exists and os.path.exists(split_path):
        # if the results exist and overwrite is false, load the cv splits.
        # The models will be trained on the existing cv splits.
        print(f"Loading existing cv splits from {split_path}")
        response_data.load_splits(path=split_path)
    else:
        # if the results do not exist, create the cv splits
        print(f"Creating cv splits at {split_path}")

        os.makedirs(result_path, exist_ok=True)

        response_data.remove_nan_responses()
        # if this line changes, also change it in pipeline: cv_split.py
        response_data.split_dataset(
            n_cv_splits=n_cv_splits,
            mode=test_mode,
            split_validation=True,
            validation_ratio=0.1,
            random_state=42,
        )
        response_data.save_splits(path=split_path)

    model_list = make_model_list(models + baselines, response_data)
    for model_name in model_list.keys():
        print(f"Running {model_name}")
        model_name, drug_id = get_model_name_and_drug_id(model_name)

        model_class = MODEL_FACTORY[model_name]
        if model_class in baselines:
            print("- Only Baseline Tests -")
            is_baseline = True
        else:
            print("- Full Test -")
            is_baseline = False

        predictions_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=drug_id,
            result_path=result_path,
            suffix="predictions",
        )
        hpam_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=drug_id,
            result_path=result_path,
            suffix="best_hpams",
        )
        parent_dir = os.path.dirname(predictions_path)

        model_hpam_set = model_class.get_hyperparameter_set()

        if response_data.cv_splits is None:
            raise ValueError("No cv splits found.")

        for split_index, split in enumerate(response_data.cv_splits):
            print(f"################# FOLD {split_index+1}/{len(response_data.cv_splits)} " f"#################")

            prediction_file = os.path.join(predictions_path, f"predictions_split_{split_index}.csv")

            hpam_filename = f"best_hpams_split_{split_index}.json"
            hpam_save_path = os.path.join(hpam_path, hpam_filename)

            (
                train_dataset,
                validation_dataset,
                early_stopping_dataset,
                test_dataset,
            ) = get_datasets_from_cv_split(split, model_class, model_name, drug_id)

            model = model_class()

            if not os.path.isfile(
                prediction_file
            ):  # if this split has not been run yet (or for a single drug model, this drug_id)

                tuning_inputs = {
                    "model": model,
                    "train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "early_stopping_dataset": early_stopping_dataset,
                    "hpam_set": model_hpam_set,
                    "response_transformation": response_transformation,
                    "metric": metric,
                    "path_data": path_data,
                }

                if multiprocessing:
                    tuning_inputs["ray_path"] = os.path.abspath(os.path.join(result_path, "raytune"))
                    best_hpams = hpam_tune_raytune(**tuning_inputs)
                else:
                    best_hpams = hpam_tune(**tuning_inputs)

                print(f"Best hyperparameters: {best_hpams}")
                print("Training model on full train and validation set to predict test set")
                # save best hyperparameters as json
                with open(
                    hpam_save_path,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(best_hpams, f)

                train_dataset.add_rows(validation_dataset)  # use full train val set data for final training
                train_dataset.shuffle(random_state=42)

                test_dataset = train_and_predict(
                    model=model,
                    hpams=best_hpams,
                    path_data=path_data,
                    train_dataset=train_dataset,
                    prediction_dataset=test_dataset,
                    early_stopping_dataset=(early_stopping_dataset if model.early_stopping else None),
                    response_transformation=response_transformation,
                )

                for cross_study_dataset in cross_study_datasets:
                    print(f"Cross study prediction on {cross_study_dataset.dataset_name}")
                    cross_study_dataset.remove_nan_responses()
                    cross_study_prediction(
                        dataset=cross_study_dataset,
                        model=model,
                        test_mode=test_mode,
                        train_dataset=train_dataset,
                        path_data=path_data,
                        early_stopping_dataset=(early_stopping_dataset if model.early_stopping else None),
                        response_transformation=response_transformation,
                        path_out=parent_dir,
                        split_index=split_index,
                        single_drug_id=(drug_id if model_name in SINGLE_DRUG_MODEL_FACTORY else None),
                    )

                test_dataset.save(prediction_file)
            else:
                print(f"Split {split_index} already exists. Skipping.")
                with open(
                    hpam_save_path,
                    encoding="utf-8",
                ) as f:
                    best_hpams = json.load(f)
            if not is_baseline:
                if randomization_mode is not None:
                    print(f"Randomization tests for {model_class.get_model_name()}")
                    # if this line changes, it also needs to be changed in pipeline:
                    # randomization_split.py
                    randomization_test_views = get_randomization_test_views(
                        model=model, randomization_mode=randomization_mode
                    )
                    randomization_test(
                        randomization_test_views=randomization_test_views,
                        model=model,
                        hpam_set=best_hpams,
                        path_data=path_data,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        early_stopping_dataset=(early_stopping_dataset if model.early_stopping else None),
                        path_out=parent_dir,
                        split_index=split_index,
                        randomization_type=randomization_type,
                        response_transformation=response_transformation,
                    )
                if n_trials_robustness > 0:
                    print(f"Robustness test for {model_class.get_model_name()}")
                    robustness_test(
                        n_trials=n_trials_robustness,
                        model=model,
                        hpam_set=best_hpams,
                        path_data=path_data,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        early_stopping_dataset=(early_stopping_dataset if model.early_stopping else None),
                        path_out=parent_dir,
                        split_index=split_index,
                        response_transformation=response_transformation,
                    )
    consolidate_single_drug_model_predictions(
        models=models,
        n_cv_splits=n_cv_splits,
        results_path=result_path,
        cross_study_datasets=cross_study_datasets,
        randomization_mode=randomization_mode,
        n_trials_robustness=n_trials_robustness,
        out_path=result_path,
    )
    print("Done!")


@pipeline_function
def consolidate_single_drug_model_predictions(
    models: list[type[DRPModel]],
    n_cv_splits: int,
    results_path: str,
    cross_study_datasets: list[DrugResponseDataset],
    randomization_mode: Optional[list[str]] = None,
    n_trials_robustness: int = 0,
    out_path: str = "",
) -> None:
    """
    Consolidate single drug model predictions into a single file.

    :param models: list of model classes to compare, e.g., [SimpleNeuralNetwork, RandomForest]
    :param n_cv_splits: number of cross-validation splits, e.g., 5
    :param results_path: path to the results directory, e.g., results/
    :param cross_study_datasets: list of cross-study datasets, e.g., [CCLE, GDSC1]
    :param randomization_mode: list of randomization modes, e.g., ["SVCC", "SVRC"]
    :param n_trials_robustness: number of robustness trials, e.g., 10
    :param out_path: for the package, this is the same as results_path. For the pipeline, this is empty because it
        will be stored in the work directory.
    """
    for model in models:
        if model.get_model_name() in SINGLE_DRUG_MODEL_FACTORY:

            model_instance = MODEL_FACTORY[model.get_model_name()]()
            model_path = os.path.join(results_path, model.get_model_name())
            out_path = os.path.join(out_path, model.get_model_name())
            os.makedirs(os.path.join(out_path, "predictions"), exist_ok=True)
            if cross_study_datasets:
                os.makedirs(os.path.join(out_path, "cross_study"), exist_ok=True)
            if randomization_mode:
                os.makedirs(os.path.join(out_path, "randomization"), exist_ok=True)
            if n_trials_robustness:
                os.makedirs(os.path.join(out_path, "robustness"), exist_ok=True)

            for split in range(n_cv_splits):

                # Collect predictions for drugs across all scenarios (main, cross_study, robustness, randomization)
                predictions: Any = {
                    "main": [],
                    "cross_study": {},
                    "robustness": {},
                    "randomization": {},
                }
                # list all dirs in model_path/drugs
                drugs = [
                    d
                    for d in os.listdir(os.path.join(model_path, "drugs"))
                    if os.path.isdir(os.path.join(model_path, "drugs", d))
                ]
                for drug in drugs:
                    single_drug_prediction_path = os.path.join(model_path, "drugs", drug)

                    # Main predictions
                    predictions["main"].append(
                        pd.read_csv(
                            os.path.join(
                                single_drug_prediction_path,
                                "predictions",
                                f"predictions_split_{split}.csv",
                            ),
                            index_col=0,
                        )
                    )

                    # Cross study predictions
                    for cross_study_dataset in cross_study_datasets:
                        cross_study_prediction_path = os.path.join(single_drug_prediction_path, "cross_study")
                        f = f"cross_study_{cross_study_dataset.dataset_name}_split_{split}.csv"
                        if cross_study_dataset.dataset_name not in predictions["cross_study"]:
                            predictions["cross_study"][cross_study_dataset.dataset_name] = []
                        predictions["cross_study"][cross_study_dataset.dataset_name].append(
                            pd.read_csv(
                                os.path.join(cross_study_prediction_path, f),
                                index_col=0,
                            )
                        )

                    # Robustness predictions
                    for trial in range(n_trials_robustness):
                        robustness_path = os.path.join(single_drug_prediction_path, "robustness")
                        f = f"robustness_{trial+1}_split_{split}.csv"
                        if trial not in predictions["robustness"]:
                            predictions["robustness"][trial] = []
                        predictions["robustness"][trial].append(
                            pd.read_csv(os.path.join(robustness_path, f), index_col=0)
                        )

                    # Randomization predictions
                    if randomization_mode is not None:
                        randomization_test_views = get_randomization_test_views(
                            model=model_instance,
                            randomization_mode=randomization_mode,
                        )
                        for view in randomization_test_views:
                            randomization_path = os.path.join(single_drug_prediction_path, "randomization")
                            f = f"randomization_{view}_split_{split}.csv"
                            if view not in predictions["randomization"]:
                                predictions["randomization"][view] = []
                            predictions["randomization"][view].append(
                                pd.read_csv(
                                    os.path.join(randomization_path, f),
                                    index_col=0,
                                )
                            )

                # Save the consolidated predictions
                pd.concat(predictions["main"], axis=0).to_csv(
                    os.path.join(
                        out_path,
                        "predictions",
                        f"predictions_split_{split}.csv",
                    )
                )

                for dataset_name, dataset_predictions in predictions["cross_study"].items():
                    pd.concat(dataset_predictions, axis=0).to_csv(
                        os.path.join(
                            out_path,
                            "cross_study",
                            f"cross_study_{dataset_name}_split_{split}.csv",
                        )
                    )

                for trial, trial_predictions in predictions["robustness"].items():
                    pd.concat(trial_predictions, axis=0).to_csv(
                        os.path.join(
                            out_path,
                            "robustness",
                            f"robustness_{trial+1}_split_{split}.csv",
                        )
                    )

                for view, view_predictions in predictions["randomization"].items():
                    pd.concat(view_predictions, axis=0).to_csv(
                        os.path.join(
                            out_path,
                            "randomization",
                            f"randomization_{view}_split_{split}.csv",
                        )
                    )


def load_features(
    model: DRPModel, path_data: str, dataset: DrugResponseDataset
) -> tuple[FeatureDataset, Optional[FeatureDataset]]:
    """
    Load and reduce cell line and drug features for a given dataset.

    :param model: model to use, e.g., SimpleNeuralNetwork
    :param path_data: path to the data directory, e.g., data/
    :param dataset: dataset to load features for, e.g., GDSC2
    :returns: tuple of cell line and, potentially, drug features
    """
    cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=dataset.dataset_name)
    drug_features = model.load_drug_features(data_path=path_data, dataset_name=dataset.dataset_name)
    return cl_features, drug_features


@pipeline_function
def cross_study_prediction(
    dataset: DrugResponseDataset,
    model: DRPModel,
    test_mode: str,
    train_dataset: DrugResponseDataset,
    path_data: str,
    early_stopping_dataset: Optional[DrugResponseDataset],
    response_transformation: Optional[TransformerMixin],
    path_out: str,
    split_index: int,
    single_drug_id: Optional[str] = None,
) -> None:
    """
    Run the drug response prediction experiment on a cross-study dataset to assess the generalizability of the model.

    :param dataset: cross-study dataset, e.g., GDSC1 if trained on GDSC2
    :param model: model to use, e.g, SimpleNeuralNetwork
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out,
        leave-drug-out)
    :param train_dataset: training dataset, e.g., GDSC2
    :param path_data: path to the data directory, e.g., data/
    :param early_stopping_dataset: early stopping dataset
    :param response_transformation: normalizer to use for the response data, e.g., StandardScaler
    :param path_out: path to the output directory, e.g., results/
    :param split_index: index of the split
    :param single_drug_id: drug id to use for single drug models None for global models
    :raises ValueError: if feature loading fails or if the test mode is invalid
    """
    dataset = dataset.copy()
    os.makedirs(os.path.join(path_out, "cross_study"), exist_ok=True)
    if response_transformation:
        dataset.transform(response_transformation)

    # load features
    try:
        cl_features, drug_features = load_features(model, path_data, dataset)
    except ValueError as e:
        warnings.warn(str(e), stacklevel=2)
        return

    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None

    drugs_to_keep: Optional[np.ndarray] = None
    if single_drug_id is not None:
        drugs_to_keep = np.array([single_drug_id])
    elif drug_features is not None:
        drugs_to_keep = drug_features.identifiers

    print(
        f"Reducing cross study dataset ... feature data available for "
        f'{len(cell_lines_to_keep) if cell_lines_to_keep is not None else "all"} cell lines '
        f'and {len(drugs_to_keep)if drugs_to_keep is not None else "all"} drugs.'
    )

    # making sure there are no missing features. Only keep cell lines and drugs for which we have
    # a feature representation
    dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if early_stopping_dataset is not None:
        train_dataset.add_rows(early_stopping_dataset)
    # remove rows which overlap in the training. depends on the test mode
    if test_mode == "LPO":
        train_pairs = {
            f"{cl}_{drug}" for cl, drug in zip(train_dataset.cell_line_ids, train_dataset.drug_ids, strict=True)
        }
        dataset_pairs = [f"{cl}_{drug}" for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids, strict=True)]

        dataset.remove_rows(np.array([i for i, pair in enumerate(dataset_pairs) if pair in train_pairs]))
    elif test_mode == "LCO":
        train_cell_lines = train_dataset.cell_line_ids
        dataset.reduce_to(
            cell_line_ids=np.setdiff1d(dataset.cell_line_ids, train_cell_lines),
            drug_ids=None,
        )
    elif test_mode == "LDO":
        train_drugs = train_dataset.drug_ids
        dataset.reduce_to(
            cell_line_ids=None,
            drug_ids=np.setdiff1d(dataset.drug_ids, train_drugs),
        )
    else:
        raise ValueError(f"Invalid test mode: {test_mode}. Choose from LPO, LCO, LDO")
    if len(dataset) > 0:
        dataset.shuffle(random_state=42)
        dataset._predictions = model.predict(
            cell_line_ids=dataset.cell_line_ids,
            drug_ids=dataset.drug_ids,
            cell_line_input=cl_features,
            drug_input=drug_features,
        )
        if response_transformation:
            dataset._response = response_transformation.inverse_transform(dataset.response)
    else:
        dataset._predictions = np.array([])
    dataset.save(
        os.path.join(
            path_out,
            "cross_study",
            f"cross_study_{dataset.dataset_name}_split_{split_index}.csv",
        )
    )


@pipeline_function
def get_randomization_test_views(model: DRPModel, randomization_mode: list[str]) -> dict[str, list[str]]:
    """
    Get the views to use for the randomization tests.

    * For SVCC, a single cell line view (e.g., gene expression) is held constant while the others are randomized.
    * For SVCD, a single drug view (e.g., fingerprints) is held constant while the others are randomized.
    * For SVRC, a single cell line view is randomized while the others are held constant.
    * For SVRD, a single drug view is randomized while the others are held constant.

    :param model: model to use, e.g., SimpleNeuralNetwork
    :param randomization_mode: list of randomization modes to do, e.g., ["SVCC", "SVRC"]
    :returns: dictionary of randomization test views
    """
    cell_line_views = model.cell_line_views
    drug_views = model.drug_views
    randomization_test_views = {}
    if "SVCC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVCC_{view}"] = [v for v in cell_line_views if v != view]
    if "SVCD" in randomization_mode:
        for view in drug_views:
            randomization_test_views[f"SVCD_{view}"] = [v for v in drug_views if v != view]
    if "SVRC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVRC_{view}"] = [view]
    if "SVRD" in randomization_mode:
        for view in drug_views:
            randomization_test_views[f"SVRD_{view}"] = [view]

    return randomization_test_views


def robustness_test(
    n_trials: int,
    model: DRPModel,
    hpam_set: dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    path_out: str,
    split_index: int,
    response_transformation: Optional[TransformerMixin] = None,
):
    """
    Run robustness tests for the given model and dataset.

    This will run the model n times with different random seeds to get a distribution of the results.

    :param n_trials: number of trials to run
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param path_data: path to the data directory
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param path_out: path to the output directory
    :param split_index: index of the split
    :param response_transformation: sklearn.preprocessing scaler like StandardScaler or MinMaxScaler to use to scale
        the target
    """
    robustness_test_path = os.path.join(path_out, "robustness")
    os.makedirs(robustness_test_path, exist_ok=True)
    for trial in range(n_trials):
        print(f"Running robustness test trial {trial+1}/{n_trials}")
        trial_file = os.path.join(
            robustness_test_path,
            f"robustness_{trial+1}_split_{split_index}.csv",
        )
        if not os.path.isfile(trial_file):
            robustness_train_predict(
                trial=trial,
                trial_file=trial_file,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                early_stopping_dataset=early_stopping_dataset,
                model=model,
                hpam_set=hpam_set,
                path_data=path_data,
                response_transformation=response_transformation,
            )


@pipeline_function
def robustness_train_predict(
    trial: int,
    trial_file: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    model: DRPModel,
    hpam_set: dict,
    path_data: str,
    response_transformation: Optional[TransformerMixin] = None,
):
    """
    Train and predict for the robustness test.

    :param trial: trial number
    :param trial_file: file to save the results to
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param path_data: path to the data directory, e.g., data/
    :param response_transformation: sklearn.preprocessing scaler like StandardScaler or MinMaxScaler to use to scale
    """
    train_dataset.shuffle(random_state=trial)
    test_dataset.shuffle(random_state=trial)
    if early_stopping_dataset is not None:
        early_stopping_dataset.shuffle(random_state=trial)
    test_dataset = train_and_predict(
        model=model,
        hpams=hpam_set,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=test_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
    )
    test_dataset.save(trial_file)


def randomization_test(
    randomization_test_views: dict[str, list[str]],
    model: DRPModel,
    hpam_set: dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    path_out: str,
    split_index: int,
    randomization_type: str = "permutation",
    response_transformation=Optional[TransformerMixin],
) -> None:
    """
    Run randomization tests for the given model and dataset.

    :param randomization_test_views: views to use for the randomization tests.
        Key is the name of the randomization test and the value is a list of views to randomize
        e.g. {"randomize_genomics": ["copy_number_var", "mutation"],
        "methylation_only": ["gene_expression", "copy_number_var", "mutation"]}"
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param path_data: path to the data directory
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param path_out: path to the output directory
    :param split_index: index of the split
    :param randomization_type: type of randomization to use. Choose from "permutation", "invariant".
        Default is "permutation" which permutes the features over the instances, keeping the
        distribution of the features the same but dissolving the relationship to the target.
        invariant randomization is done in a way that a key characteristic of the feature is preserved.
        In case of matrices, this is the mean and standard deviation of the feature view for this
        instance, for networks it is the degree distribution.
    :param response_transformation: sklearn.preprocessing scaler like StandardScaler or MinMaxScaler
        to use to scale the target
    """
    for test_name, views in randomization_test_views.items():
        randomization_test_path = os.path.join(path_out, "randomization")
        os.makedirs(randomization_test_path, exist_ok=True)

        randomization_test_file = os.path.join(
            randomization_test_path,
            f"randomization_{test_name}_split_{split_index}.csv",
        )
        if not os.path.isfile(randomization_test_file):  # if this splits test has not been run yet
            for view in views:
                print(f"Randomizing view {view} for randomization test {test_name} ...")
                randomize_train_predict(
                    view=view,
                    test_name=test_name,
                    randomization_type=randomization_type,
                    randomization_test_file=randomization_test_file,
                    model=model,
                    hpam_set=hpam_set,
                    path_data=path_data,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    early_stopping_dataset=early_stopping_dataset,
                    response_transformation=response_transformation,
                )
        else:
            print(f"Randomization test {test_name} already exists. Skipping.")


@pipeline_function
def randomize_train_predict(
    view: str,
    test_name: str,
    randomization_type: str,
    randomization_test_file: str,
    model: DRPModel,
    hpam_set: dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    response_transformation: Optional[TransformerMixin],
) -> None:
    """
    Randomize the features for a given view and run the model.

    :param view: view to randomize, e.g., gene_expression
    :param test_name: name of the randomization test, e.g., SVRC_gene_expression
    :param randomization_type: type of randomization to use, e.g., permutation
    :param randomization_test_file: file to save the results to
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param path_data: path to the data directory
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param response_transformation: sklearn.preprocessing scaler like StandardScaler or MinMaxScaler to use to scale
    """
    cl_features, drug_features = load_features(model, path_data, train_dataset)

    # Handle case where both features are None early on
    if cl_features is None and drug_features is None:
        warnings.warn(
            "Both cl_features and drug_features are None. Skipping randomization test.",
            stacklevel=2,
        )
        return

    # Check if view is in either feature set, if not, warn and skip
    if (cl_features is not None and view not in cl_features.view_names) and (
        drug_features is not None and view not in drug_features.view_names
    ):
        warnings.warn(
            f"View {view} not found in features. Skipping randomization test {test_name} which includes this view.",
            stacklevel=2,
        )
        return

    cl_features_rand: Optional[FeatureDataset] = None
    if cl_features is not None:
        cl_features_rand = cl_features.copy()
        cl_features_rand.randomize_features(view, randomization_type=randomization_type)  # type: ignore[union-attr]

    drug_features_rand: Optional[FeatureDataset] = None
    if drug_features is not None:
        drug_features_rand = drug_features.copy()
        drug_features_rand.randomize_features(view, randomization_type=randomization_type)  # type: ignore[union-attr]

    test_dataset_rand = train_and_predict(
        model=model,
        hpams=hpam_set,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=test_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
        cl_features=cl_features_rand,
        drug_features=drug_features_rand,
    )
    test_dataset_rand.save(randomization_test_file)


def split_early_stopping(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Split the validation dataset into a validation and early stopping dataset.

    :param validation_dataset: validation dataset
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :returns: tuple of validation and early stopping datasets
    """
    validation_dataset.shuffle(random_state=42)
    cv_v = validation_dataset.split_dataset(
        n_cv_splits=4,
        mode=test_mode,
        split_validation=False,
        random_state=42,
    )
    # take the first fold of a 4 cv as the split ie. 3/4 for validation and 1/4 for early stopping
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


@pipeline_function
def train_and_predict(
    model: DRPModel,
    hpams: dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    cl_features: Optional[FeatureDataset] = None,
    drug_features: Optional[FeatureDataset] = None,
) -> DrugResponseDataset:
    """
    Train the model and predict the response for the prediction dataset.

    :param model: model to use, e.g., SimpleNeuralNetwork
    :param hpams: hyperparameters to use
    :param path_data: path to the data directory, e.g., data/
    :param train_dataset: training dataset
    :param prediction_dataset: prediction dataset
    :param early_stopping_dataset: early stopping dataset, optional
    :param response_transformation: normalizer to use for the response data, e.g., StandardScaler
    :param cl_features: cell line features
    :param drug_features: drug features
    :returns: prediction dataset with predictions
    :raises ValueError: if train_dataset does not have a dataset_name
    """
    model.build_model(hyperparameters=hpams)
    if train_dataset.dataset_name is None:
        raise ValueError("train_dataset must have a dataset_name")
    if cl_features is None:
        print("Loading cell line features ...")
        cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    if drug_features is None:
        print("Loading drug features ...")
        drug_features = model.load_drug_features(data_path=path_data, dataset_name=train_dataset.dataset_name)

    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None

    # making sure there are no missing features:
    len_train_before = len(train_dataset)
    len_pred_before = len(prediction_dataset)
    print(f"Number of cell lines in features: {len(cell_lines_to_keep)}")
    if drugs_to_keep is not None:
        print(f"Number of drugs in features: {len(drugs_to_keep)}")
    print(f"Number of cell lines in train dataset: {len(np.unique(train_dataset.cell_line_ids))}")
    print(f"Number of drugs in train dataset: {len(np.unique(train_dataset.drug_ids))}")

    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    prediction_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}, because of missing features")
    print(
        f"Reduced prediction dataset from {len_pred_before} to {len(prediction_dataset)}, because of missing features"
    )

    if early_stopping_dataset is not None:
        len_es_before = len(early_stopping_dataset)
        early_stopping_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        print(f"Reduced early stopping dataset from {len_es_before} to {len(early_stopping_dataset)}")

    if response_transformation:
        train_dataset.fit_transform(response_transformation)
        if early_stopping_dataset is not None:
            early_stopping_dataset.transform(response_transformation)
        prediction_dataset.transform(response_transformation)

    print("Training model ...")
    model.train(
        output=train_dataset,
        cell_line_input=cl_features,
        drug_input=drug_features,
        output_earlystopping=early_stopping_dataset,
    )
    if len(prediction_dataset) > 0:
        prediction_dataset._predictions = model.predict(
            cell_line_ids=prediction_dataset.cell_line_ids,
            drug_ids=prediction_dataset.drug_ids,
            cell_line_input=cl_features,
            drug_input=drug_features,
        )

        if response_transformation:
            prediction_dataset.inverse_transform(response_transformation)
    else:
        prediction_dataset._predictions = np.array([])

    return prediction_dataset


def train_and_evaluate(
    model: DRPModel,
    hpams: dict[str, list],
    path_data: str,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
) -> dict[str, float]:
    """
    Train and evaluate the model, i.e., call train_and_predict() and then evaluate().

    :param model: model to use
    :param hpams: hyperparameters to use
    :param path_data: path to the data directory
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param early_stopping_dataset: early stopping dataset
    :param response_transformation: normalizer to use for the response data
    :param metric: metric to evaluate the model on
    :returns: dictionary of the evaluation results, e.g., {"RMSE": 0.1}
    """
    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
    )
    return evaluate(validation_dataset, metric=[metric])


def hpam_tune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: list[dict],
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "RMSE",
    path_data: str = "data",
) -> dict:
    """
    Tune the hyperparameters for the given model in an iterative manner.

    :param model: model to use
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param hpam_set: hyperparameters to tune
    :param early_stopping_dataset: early stopping dataset
    :param response_transformation: normalizer to use for the response data
    :param metric: metric to evaluate which model is the best
    :param path_data: path to the data directory, e.g., data/
    :returns: best hyperparameters
    :raises AssertionError: if hpam_set is empty
    """
    if len(hpam_set) == 0:
        raise AssertionError("hpam_set must contain at least one hyperparameter configuration")
    if len(hpam_set) == 1:
        return hpam_set[0]

    best_hyperparameters = None
    mode = get_mode(metric)
    best_score = float("inf") if mode == "min" else float("-inf")
    for hyperparameter in hpam_set:
        print(f"Training model with hyperparameters: {hyperparameter}")
        score = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
            path_data=path_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation,
        )[metric]

        if np.isnan(score):
            continue

        if (mode == "min" and score < best_score) or (mode == "max" and score > best_score):
            print(f"current best {metric} score: {np.round(score, 3)}")
            best_score = score
            best_hyperparameters = hyperparameter

    if best_hyperparameters is None:
        warnings.warn("all hpams lead to NaN respone. using last hpam combination.", stacklevel=2)
        best_hyperparameters = hyperparameter

    return best_hyperparameters


def hpam_tune_raytune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    hpam_set: list[dict],
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "RMSE",
    ray_path: str = "raytune",
    path_data: str = "data",
) -> dict:
    """
    Tune the hyperparameters for the given model using raytune.

    :param model: model to use
    :param train_dataset: training dataset
    :param validation_dataset: validation dataset
    :param early_stopping_dataset: early stopping dataset
    :param hpam_set: hyperparameters to tune
    :param response_transformation: normalizer to use for the response data
    :param metric: metric to evaluate which model is the best
    :param ray_path: path to the raytune directory
    :param path_data: path to the data directory, e.g., data/
    :returns: best hyperparameters
    """
    if len(hpam_set) == 1:
        return hpam_set[0]
    ray.init(_temp_dir=os.path.join(os.path.expanduser("~"), "raytmp"))
    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}  # TODO make this user defined
    else:
        resources_per_trial = {"cpu": 1}  # TODO make this user defined
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            path_data=path_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation,
        ),
        config=tune.grid_search(hpam_set),
        mode="min",
        num_samples=5,
        resources_per_trial=resources_per_trial,
        chdir_to_trial_dir=False,
        verbose=0,
        storage_path=ray_path,
    )

    mode = get_mode(metric)
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    return best_config


@pipeline_function
def make_model_list(models: list[type[DRPModel]], response_data: DrugResponseDataset) -> dict[str, str]:
    """
    Make a list of models to evaluate: if it is a single drug model, add the drug id to the model name.

    :param models: list of models to evaluate
    :param response_data: response data, needed to get the unique drugs for single drug models
    :returns: dictionary of model names: model class, e.g., {"SimpleNeuralNetwork": "SimpleNeuralNetwork",
        "MOLIR.Afatinib": "MOLIR"}
    """
    model_list = {}
    unique_drugs = np.unique(response_data.drug_ids)
    for model in models:
        if model.is_single_drug_model:
            for drug in unique_drugs:
                model_list[f"{model.get_model_name()}.{drug}"] = model.get_model_name()
        else:
            model_list[model.get_model_name()] = model.get_model_name()
    return model_list


@pipeline_function
def get_model_name_and_drug_id(model_name: str) -> tuple[str, Optional[str]]:
    """
    Get the model name and drug id from the model name.

    :param model_name: model name, e.g., SimpleNeuralNetwork or MOLIR.Afatinib
    :returns: tuple of model name and, potentially drug id if it is a single drug model
    :raises AssertionError: if the model name is not found in the model factory
    """
    if model_name in MULTI_DRUG_MODEL_FACTORY:
        return model_name, None
    else:
        name_split = model_name.split(".")
        model_name = name_split[0]
        if model_name not in SINGLE_DRUG_MODEL_FACTORY:
            raise AssertionError(
                f"Model {model_name} not found in MODEL_FACTORY or SINGLE_DRUG_MODEL_FACTORY. "
                "Please add the model to the factory."
            )
        drug_id = name_split[1]

        return model_name, drug_id


@pipeline_function
def get_datasets_from_cv_split(
    split: dict[str, DrugResponseDataset], model_class: type[DRPModel], model_name: str, drug_id: Optional[str] = None
) -> tuple[DrugResponseDataset, DrugResponseDataset, Optional[DrugResponseDataset], DrugResponseDataset]:
    """
    Get train, validation, (early stopping), and test datasets from the CV split.

    :param split: dictionary of the CV split
    :param model_class: model class
    :param model_name: model name
    :param drug_id: drug id for single drug models
    :returns: tuple of train, validation, (early stopping), and test datasets
    """
    train_dataset = split["train"]
    validation_dataset = split["validation"]
    test_dataset = split["test"]

    if model_class.early_stopping:
        validation_dataset = split["validation_es"]
        early_stopping_dataset = split["early_stopping"]
    else:
        early_stopping_dataset = None

    if model_name in SINGLE_DRUG_MODEL_FACTORY.keys():
        output_mask = train_dataset.drug_ids == drug_id
        train_cp = train_dataset.copy()
        train_cp.mask(output_mask)
        validation_mask = validation_dataset.drug_ids == drug_id
        val_cp = validation_dataset.copy()
        val_cp.mask(validation_mask)
        test_mask = test_dataset.drug_ids == drug_id
        test_cp = test_dataset.copy()
        test_cp.mask(test_mask)
        if early_stopping_dataset is not None:
            es_mask = early_stopping_dataset.drug_ids == drug_id
            es_cp = early_stopping_dataset.copy()
            es_cp.mask(es_mask)
            return train_cp, val_cp, es_cp, test_cp
        return train_cp, val_cp, None, test_cp

    return (
        train_dataset,
        validation_dataset,
        early_stopping_dataset,
        test_dataset,
    )


@pipeline_function
def generate_data_saving_path(model_name, drug_id, result_path, suffix) -> str:
    """
    Generate a path to save data to.

    For single drug models, the path is result_path/model_name/drugs/drug_id/suffix.
    For all others, it is result_path/model_name/suffix.
    :param model_name: model name
    :param drug_id: drug id
    :param result_path: path to the results directory
    :param suffix: suffix to add to the path, e.g., "predictions", "best_hpams", "randomization", "robustness"
    :returns: path to save data to
    """
    is_single_drug_model = model_name in SINGLE_DRUG_MODEL_FACTORY
    if is_single_drug_model:
        model_path = os.path.join(result_path, model_name, "drugs", drug_id, suffix)
    else:
        model_path = os.path.join(result_path, model_name, suffix)
    os.makedirs(model_path, exist_ok=True)
    return model_path

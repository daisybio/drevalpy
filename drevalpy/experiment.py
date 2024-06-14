import json
from typing import Dict, List, Optional, Tuple, Type
import warnings
from drevalpy.utils import handle_overwrite
from .datasets.dataset import DrugResponseDataset, FeatureDataset
from .evaluation import evaluate, get_mode
from .models.drp_model import CompositeDrugModel, DRPModel, SingleDrugModel
import numpy as np
import os
import ray
import torch
from ray import tune
from sklearn.base import TransformerMixin


def drug_response_experiment(
    models: List[Type[DRPModel]],
    response_data: DrugResponseDataset,
    baselines: Optional[List[Type[DRPModel]]] = None,
    response_transformation: Optional[TransformerMixin] = None,
    run_id: str = "",
    test_mode: str = "LPO",
    metric: str = "rmse",
    n_cv_splits: int = 5,
    multiprocessing: bool = False,
    randomization_mode: Optional[List[str]] = None,
    randomization_type: str = "permutation",
    cross_study_datasets: Optional[List[DrugResponseDataset]] = None,
    n_trials_robustness: int = 0,
    path_out: str = "results/",
    overwrite: bool = False,
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
    :param randomization_mode: list of randomization modes to do.
        Modes: SVCC, SVRC, SVCD, SVRD
        Can be a list of randomization tests e.g. 'SVCC SVCD'. Default is None, which means no randomization tests are run.
        SVCC: Single View Constant for Cell Lines: in this mode, one experiment is done for every cell line view the model uses (e.g. gene expression, mutation, ..).
        For each experiment one cell line view is held constant while the others are randomized.
        SVRC Single View Random for Cell Lines: in this mode, one experiment is done for every cell line view the model uses (e.g. gene expression, mutation, ..).
        For each experiment one cell line view is randomized while the others are held constant.
        SVCD: Single View Constant for Drugs: in this mode, one experiment is done for every drug view the model uses (e.g. fingerprints, target_information, ..).
        For each experiment one drug view is held constant while the others are randomized.
        SVRD: Single View Random for Drugs: in this mode, one experiment is done for every drug view the model uses (e.g. gene expression, target_information, ..).
        For each experiment one drug view is randomized while the others are held constant.
    :param randomization_type: type of randomization to use. Choose from "gaussian", "zeroing", "permutation". Default is "permutation"
            "gaussian": replace the features with random values sampled from a gaussian distribution with the same mean and standard deviation
            "zeroing": replace the features with zeros
            "permutation": permute the features over the instances, keeping the distribution of the features the same but dissolving the relationship to the target
    :param n_trials_robustness: number of trials to run for the robustness test. The robustness test is a test where models are retrained multiple tiems with varying seeds. Default is 0, which means no robustness test is run.
    :param path_out: path to the output directory
    :param run_id: identifier to save the results
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param overwrite: whether to overwrite existing results

    :return: None
    """
    if baselines is None:
        baselines = []
    cross_study_datasets = cross_study_datasets or []
    result_path = os.path.join(path_out, run_id, test_mode)

    # if results exists, delete them if overwrite is true
    handle_overwrite(result_path, overwrite)

    for model_class in models + baselines:
        if model_class in baselines:
            print(f"Running baseline model {model_class.model_name}")
            is_baseline = True
        else:
            print(f"Running model {model_class.model_name}")
            is_baseline = False

        model_path = os.path.join(result_path, model_class.model_name)
        handle_overwrite(model_path, overwrite)
        predictions_path = os.path.join(model_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        if randomization_mode is not None:
            randomization_test_path = os.path.join(model_path, "randomization_tests")
            os.makedirs(randomization_test_path)

        model_hpam_set = model_class.get_hyperparameter_set()

        response_data.remove_nan_responses()

        response_data.split_dataset(
            n_cv_splits=n_cv_splits,
            mode=test_mode,
            split_validation=True,
            validation_ratio=0.1,
            random_state=42,
        )

        for split_index, split in enumerate(response_data.cv_splits):
            prediction_file = os.path.join(
                predictions_path, f"test_dataset_{test_mode}_split_{split_index}.csv"
            )
            # if model_class.early_stopping is true then we split the validation set into a validation and early stopping set
            train_dataset = split["train"]
            validation_dataset = split["validation"]
            test_dataset = split["test"]

            if model_class.early_stopping:
                validation_dataset, early_stopping_dataset = split_early_stopping(
                    validation_dataset=validation_dataset, test_mode=test_mode
                )

            if issubclass(model_class, SingleDrugModel):
                model = CompositeDrugModel(target="IC50", base_model=model_class)
            else:
                model = model_class(target="IC50")

            if not os.path.isfile(
                prediction_file
            ):  # if this split has not been run yet

                tuning_inputs = {
                    "model": model,
                    "train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "early_stopping_dataset": (
                        early_stopping_dataset if model.early_stopping else None
                    ),
                    "hpam_set": model_hpam_set,
                    "response_transformation": response_transformation,
                    "metric": metric,
                }

                if multiprocessing:
                    if Type(model) == CompositeDrugModel:
                        warnings.warn(
                            "Multiprocessing not yet supported for CompositeDrugModel."
                        )
                        best_hpams = hpam_tune_composite_model(**tuning_inputs)

                    else:
                        tuning_inputs["ray_path"] = os.path.abspath(
                            os.path.join(result_path, "raytune")
                        )
                        best_hpams = hpam_tune_raytune(**tuning_inputs)
                else:
                    if type(model) == CompositeDrugModel:
                        best_hpams = hpam_tune_composite_model(**tuning_inputs)
                    else:
                        best_hpams = hpam_tune(**tuning_inputs)

                print(f"Best hyperparameters: {best_hpams}")
                print(
                    "Training model on full train and validation set to predict test set"
                )
                # save best hyperparameters as json
                with open(
                    os.path.join(
                        predictions_path, f"best_hpams_split_{split_index}.json"
                    ),
                    "w",
                ) as f:
                    json.dump(best_hpams, f)

                train_dataset.add_rows(
                    validation_dataset
                )  # use full train val set data for final training
                train_dataset.shuffle(random_state=42)

                test_dataset = train_and_predict(
                    model=model,
                    hpams=best_hpams,
                    path_data="data",
                    train_dataset=train_dataset,
                    prediction_dataset=test_dataset,
                    early_stopping_dataset=(
                        early_stopping_dataset if model.early_stopping else None
                    ),
                    response_transformation=response_transformation,
                )

                for cross_study_dataset in cross_study_datasets:
                    cross_study_dataset.remove_nan_responses()
                    cross_study_prediction(
                        dataset=cross_study_dataset,
                        model=model,
                        test_mode=test_mode,
                        train_dataset=train_dataset,
                        path_data="data",
                        early_stopping_dataset=(
                            early_stopping_dataset if model.early_stopping else None
                        ),
                        response_transformation=response_transformation,
                        predictions_path=predictions_path,
                        split_index=split_index,
                    )

                test_dataset.save(prediction_file)
            else:
                print(f"Split {split_index} already exists. Skipping.")
                best_hpams = json.load(
                    open(
                        os.path.join(
                            predictions_path, f"best_hpams_split_{split_index}.json"
                        )
                    )
                )
            if not is_baseline:
                if randomization_mode is not None:
                    randomization_test_views = get_randomization_test_views(
                        model=model, randomization_mode=randomization_mode
                    )
                    randomization_test(
                        randomization_test_views=randomization_test_views,
                        model=model,
                        hpam_set=best_hpams,
                        path_data="data",
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        early_stopping_dataset=early_stopping_dataset,
                        path_out=randomization_test_path,
                        split_index=split_index,
                        test_mode=test_mode,
                        randomization_type=randomization_type,
                        response_transformation=response_transformation,
                    )
                if n_trials_robustness > 0:
                    robustness_test(
                        n_trials=n_trials_robustness,
                        model=model,
                        hpam_set=best_hpams,
                        path_data="data",
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        early_stopping_dataset=early_stopping_dataset,
                        path_out=model_path,
                        split_index=split_index,
                        test_mode=test_mode,
                        response_transformation=response_transformation,
                    )


def load_features(
    model: DRPModel, path_data: str, dataset: DrugResponseDataset
) -> Tuple[FeatureDataset, FeatureDataset]:
    """Load and reduce cell line and drug features for a given dataset."""
    cl_features = model.load_cell_line_features(
        data_path=path_data, dataset_name=dataset.dataset_name
    )
    drug_features = model.load_drug_features(
        data_path=path_data, dataset_name=dataset.dataset_name
    )
    return cl_features, drug_features


def cross_study_prediction(
    dataset: DrugResponseDataset,
    model: DRPModel,
    test_mode: str,
    train_dataset: DrugResponseDataset,
    path_data: str,
    early_stopping_dataset: Optional[DrugResponseDataset],
    response_transformation: Optional[TransformerMixin],
    predictions_path: str,
    split_index: int,
) -> None:
    """
    Run the drug response prediction experiment on a cross-study dataset. Save results to disc.
    :param dataset: cross-study dataset
    :param model: model to use
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param train_dataset: training dataset
    :param early_stopping_dataset: early stopping dataset
    """
    os.makedirs(os.path.join(predictions_path, "cross_study"), exist_ok=True)
    if response_transformation:
        dataset.transform(response_transformation)

    # load features
    cl_features, drug_features = load_features(model, path_data, dataset)

    cell_lines_to_remove = cl_features.identifiers if cl_features is not None else None
    drugs_to_remove = drug_features.identifiers if drug_features is not None else None

    print(
        f'Reducing cross study dataset ... feature data available for {len(cell_lines_to_remove) if cell_lines_to_remove else "all"} cell lines and {len(drugs_to_remove)if drugs_to_remove else "all"} drugs.'
    )

    # making sure there are no missing features. Only keep cell lines and drugs for which we have a feature representation
    dataset.reduce_to(cell_line_ids=cell_lines_to_remove, drug_ids=drugs_to_remove)
    if early_stopping_dataset is not None:
        train_dataset.add_rows(early_stopping_dataset)
    # remove rows which overlap in the training. depends on the test mode
    if test_mode == "LPO":
        train_pairs = set(
            [
                f"{cl}_{drug}"
                for cl, drug in zip(train_dataset.cell_line_ids, train_dataset.drug_ids)
            ]
        )
        dataset_pairs = [
            f"{cl}_{drug}" for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids)
        ]
        dataset.remove_rows(
            [i for i, pair in enumerate(dataset_pairs) if pair in train_pairs]
        )

    elif test_mode == "LCO":
        train_cell_lines = set(train_dataset.cell_line_ids)
        dataset.reduce_to(
            cell_line_ids=[
                cl for cl in dataset.cell_line_ids if cl not in train_cell_lines
            ]
        )
    elif test_mode == "LDO":
        train_drugs = set(train_dataset.drug_ids)
        dataset.reduce_to(
            drug_ids=[drug for drug in dataset.drug_ids if drug not in train_drugs]
        )
    else:
        raise ValueError(f"Invalid test mode: {test_mode}. Choose from LPO, LCO, LDO")

    dataset.shuffle(random_state=42)

    inputs = model.get_feature_matrices(
        cell_line_ids=dataset.cell_line_ids,
        drug_ids=dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )
    if type(model) == CompositeDrugModel:
        inputs["drug_ids"] = dataset.drug_ids
    dataset.predictions = model.predict(**inputs)
    if response_transformation:
        dataset.response = response_transformation.inverse_transform(dataset.response)
    dataset.save(
        os.path.join(
            predictions_path,
            "cross_study",
            f"cross_study_{dataset.dataset_name}_split_{split_index}.csv",
        )
    )


def get_randomization_test_views(
    model: DRPModel, randomization_mode: List[str]
) -> Dict[str, List[str]]:
    cell_line_views = model.cell_line_views
    drug_views = model.drug_views
    randomization_test_views = {}
    if "SVCC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVCC_{view}"] = [
                view for view in cell_line_views if view != view
            ]
    if "SVRC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVRC_{view}"] = [view]
    if "SVCD" in randomization_mode:
        for view in drug_views:
            randomization_test_views[f"SVCD_{view}"] = [
                view for view in drug_views if view != view
            ]
    if "SVRD" in randomization_mode:
        for view in drug_views:
            randomization_test_views[f"SVRD_{view}"] = [view]

    return randomization_test_views


def robustness_test(
    n_trials: int,
    model: DRPModel,
    hpam_set: Dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    path_out: str,
    split_index: int,
    test_mode: str,
    response_transformation: Optional[TransformerMixin] = None,
):
    """
    Run robustness tests for the given model and dataset (run the model n times with different random seeds to get a distribution of the results)
    :param n_trials: number of trials to run
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param path_out: path to the output directory
    :param split_index: index of the split
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param response_transformation: sklearn.preprocessing scaler like StandardScaler or MinMaxScaler to use to scale the target
    :return: None (save results to disk)
    """

    robustness_test_path = os.path.join(path_out, "robustness_test")
    os.makedirs(robustness_test_path, exist_ok=True)
    for trial in range(n_trials):
        trial_file = os.path.join(
            robustness_test_path,
            f"test_dataset_{test_mode}_split_{split_index}_{trial}.csv",
        )
        if not os.path.isfile(trial_file):
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
    randomization_test_views: Dict[str, List[str]],
    model: DRPModel,
    hpam_set: Dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    path_out: str,
    split_index: int,
    test_mode: str,
    randomization_type: str = "permutation",
    response_transformation=Optional[TransformerMixin],
) -> None:
    """
    Run randomization tests for the given model and dataset
    :param randomization_test_views: views to use for the randomization tests. Key is the name of the randomization test and the value is a list of views to randomize
            e.g. {"randomize_genomics": ["copy_number_var", "mutation"], "methylation_only": ["gene_expression", "copy_number_var", "mutation"]}"
    :param model: model to evaluate
    :param hpam_set: hyperparameters to use
    :param train_dataset: training dataset
    :param test_dataset: test dataset
    :param early_stopping_dataset: early stopping dataset
    :param path_out: path to the output directory
    :param split_index: index of the split
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param randomization_type: type of randomization to use. Choose from "gaussian", "zeroing", "permutation". Default is "permutation"
    :param response_transformation sklearn.preprocessing scaler like StandardScaler or MinMaxScaler to use to scale the target
    :return: None (save results to disk)
    """
    cl_features, drug_features = load_features(model, path_data, train_dataset)

    for test_name, views in randomization_test_views.items():
        randomization_test_path = os.path.join(path_out, test_name)
        randomization_test_file = os.path.join(
            randomization_test_path, f"test_dataset_{test_mode}_split_{split_index}.csv"
        )

        os.makedirs(randomization_test_path, exist_ok=True)
        if not os.path.isfile(
            randomization_test_file
        ):  # if this splits test has not been run yet
            for view in views:
                cl_features_rand = cl_features.copy()
                drug_features_rand = drug_features.copy()
                if view in cl_features.get_view_names():
                    cl_features_rand.randomize_features(
                        view, randomization_type=randomization_type
                    )
                elif view in drug_features.get_view_names():
                    drug_features_rand.randomize_features(
                        view, randomization_type=randomization_type
                    )
                else:
                    warnings.warn(
                        f"View {view} not found in features. Skipping randomization test {test_name} which includes this view."
                    )
                    break
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
        else:
            print(f"Randomization test {test_name} already exists. Skipping.")


def split_early_stopping(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> Tuple[DrugResponseDataset, DrugResponseDataset]:
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


def train_and_predict(
    model: DRPModel,
    hpams: Dict,
    path_data: str,
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    cl_features: Optional[FeatureDataset] = None,
    drug_features: Optional[FeatureDataset] = None,
) -> DrugResponseDataset:
    model.build_model(hyperparameters=hpams)

    if cl_features is None:
        print("Loading cell line features ...")
        cl_features = model.load_cell_line_features(
            data_path=path_data, dataset_name=train_dataset.dataset_name
        )
    if drug_features is None:
        print("Loading drug features ...")
        drug_features = model.load_drug_features(
            data_path=path_data, dataset_name=train_dataset.dataset_name
        )

    cell_lines_to_remove = cl_features.identifiers if cl_features is not None else None
    drugs_to_remove = drug_features.identifiers if drug_features is not None else None

    print(
        f'Reducing datasets ... feature data available for {len(cell_lines_to_remove) if cell_lines_to_remove else "all"} cell lines and {len(drugs_to_remove)if drugs_to_remove else "all"} drugs.'
    )

    # making sure there are no missing features:
    train_dataset.reduce_to(
        cell_line_ids=cell_lines_to_remove, drug_ids=drugs_to_remove
    )

    prediction_dataset.reduce_to(
        cell_line_ids=cell_lines_to_remove, drug_ids=drugs_to_remove
    )

    print("Constructing feature matrices ...")
    inputs = model.get_feature_matrices(
        cell_line_ids=train_dataset.cell_line_ids,
        drug_ids=train_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )
    prediction_inputs = model.get_feature_matrices(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )
    if early_stopping_dataset is not None:
        early_stopping_dataset.reduce_to(
            cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
        )
        early_stopping_inputs = model.get_feature_matrices(
            cell_line_ids=early_stopping_dataset.cell_line_ids,
            drug_ids=early_stopping_dataset.drug_ids,
            cell_line_input=cl_features,
            drug_input=drug_features,
        )
        for key in early_stopping_inputs:
            inputs[key + "_earlystopping"] = early_stopping_inputs[key]

    if response_transformation:
        train_dataset.fit_transform(response_transformation)
        early_stopping_dataset.transform(response_transformation)
        prediction_dataset.transform(response_transformation)

    print("Training model ...")
    if model.early_stopping:
        model.train(
            output=train_dataset, output_earlystopping=early_stopping_dataset, **inputs
        )
    else:
        model.train(output=train_dataset, **inputs)
    if type(model) == CompositeDrugModel:
        prediction_inputs["drug_ids"] = prediction_dataset.drug_ids
    prediction_dataset.predictions = model.predict(**prediction_inputs)

    if response_transformation:
        prediction_dataset.inverse_transform(response_transformation)

    return prediction_dataset


def train_and_evaluate(
    model: DRPModel,
    hpams: Dict[str, List],
    path_data: str,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
) -> float:
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
    hpam_set: List[Dict],
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
) -> Dict:
    best_hyperparameters = None
    mode = get_mode(metric)
    best_score = float("inf") if mode == "min" else float("-inf")
    for hyperparameter in hpam_set:
        print(f"Training model with hyperparameters: {hyperparameter}")
        score = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
            path_data="data",
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation,
        )[metric]

        if (mode == "min" and score < best_score) or (
            mode == "max" and score > best_score
        ):
            print(f"current best {metric} score: {np.round(score, 3)}")
            best_score = score
            best_hyperparameters = hyperparameter
    return best_hyperparameters


def hpam_tune_composite_model(
    model: CompositeDrugModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: List[Dict],
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
) -> Dict[str, Dict]:

    unique_drugs = list(np.unique(train_dataset.drug_ids)) + list(
        np.unique(validation_dataset.drug_ids)
    )
    # seperate best_hyperparameters for each drug
    mode = get_mode(metric)
    best_scores = {
        drug: float("inf") if mode == "min" else float("-inf") for drug in unique_drugs
    }
    best_hyperparameters = {drug: None for drug in unique_drugs}

    for hyperparameter in hpam_set:
        print(f"Training model with hyperparameters: {hyperparameter}")
        hyperparameters_per_drug = {drug: hyperparameter for drug in unique_drugs}

        validation_dataset = train_and_predict(
            model=model,
            hpams=hyperparameters_per_drug,
            path_data="data",
            train_dataset=train_dataset,
            early_stopping_dataset=early_stopping_dataset,
            prediction_dataset=validation_dataset,
            response_transformation=response_transformation,
        )

        # seperate evaluation for each drug. Each drug might have different best hyperparameters
        for drug in np.unique(validation_dataset.drug_ids):
            mask = validation_dataset.drug_ids == drug
            validation_dataset_drug = validation_dataset.copy()
            validation_dataset_drug.mask(mask)
            score = evaluate(validation_dataset_drug, metric=metric)[metric]
            if (mode == "min" and score < best_scores[drug]) or (
                mode == "max" and score > best_scores[drug]
            ):
                print(f"current best {metric} score for {drug}: { score }")
                best_scores[drug] = score
                best_hyperparameters[drug] = hyperparameter
    return best_hyperparameters


def hpam_tune_raytune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    hpam_set: List[Dict],
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
    ray_path: str = "raytune",
) -> Dict:

    ray.init(_temp_dir=os.path.join(os.path.expanduser("~"), "raytmp"))
    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}  # TODO make this user defined
    else:
        resources_per_trial = {"cpu": 1}  # TODO make this user defined
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            path_data="data",
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

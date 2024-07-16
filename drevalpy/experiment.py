import json
from typing import Dict, List, Optional, Tuple, Type
import warnings
from .datasets.dataset import DrugResponseDataset, FeatureDataset
from .evaluation import evaluate
from .models.drp_model import DRPModel
import numpy as np
import os
import shutil
import ray
import torch
from ray import tune
from sklearn.base import TransformerMixin


def drug_response_experiment(
        models: List[Type[DRPModel]],
        response_data: DrugResponseDataset,
        response_transformation: Optional[TransformerMixin] = None,
        run_id: str = "",
        test_mode: str = "LPO",
        metric: str = "rmse",
        n_cv_splits: int = 5,
        multiprocessing: bool = False,
        randomization_mode: Optional[List[str]] = None,
        randomization_type: str = "permutation",
        n_trials_robustness: int = 0,
        path_out: str = "results/",
        overwrite: bool = False,
) -> None:
    """
    Run the drug response prediction experiment. Save results to disc.
    :param models: list of model classes to compare
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

    result_path = os.path.join(path_out, run_id, test_mode)
    split_path = os.path.join(result_path, "splits")
    result_folder_exists = os.path.exists(result_path)
    if result_folder_exists and overwrite: 
        # if results exists, delete them if overwrite is True
        print(f"Overwriting existing results at {result_path}")
        shutil.rmtree(result_path)
    elif result_folder_exists and os.path.exists(split_path): 
        # if the results exist and overwrite is false, load the cv splits.
        # The models will be trained on the existing cv splits.
        print(f"Loading existing cv splits from {split_path}")
        response_data.load_splits(path=split_path)
    else:
        # if the results do not exist, create the cv splits
        print(f"Creating cv splits at {split_path}")

        os.makedirs(result_path, exist_ok=True)
        response_data.split_dataset(
                n_cv_splits=n_cv_splits,
                mode=test_mode,
                split_validation=True,
                validation_ratio=0.1,
                random_state=42,
            )
        response_data.save_splits(path=split_path)
    
    for model_class in models:

        print(f"Running model {model_class.model_name}")

        model_path = os.path.join(result_path, model_class.model_name)
        os.makedirs(model_path, exist_ok=True)
        predictions_path = os.path.join(model_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)

        if randomization_mode is not None:
            randomization_test_path = os.path.join(model_path, "randomization_tests")
            os.makedirs(randomization_test_path)

        model_hpam_set = model_class.get_hyperparameter_set()

        
        for split_index, split in enumerate(response_data.cv_splits):
            prediction_file = os.path.join(predictions_path, f"test_dataset_{test_mode}_split_{split_index}.csv")
            # if model_class.early_stopping is true then we split the validation set into a validation and early stopping set
            train_dataset = split["train"]
            validation_dataset = split["validation"]
            test_dataset = split["test"]

            if model_class.early_stopping:
                validation_dataset, early_stopping_dataset = split_early_stopping(
                    validation_dataset=validation_dataset, test_mode=test_mode
                )
            model = model_class(target="IC50")

            if not os.path.isfile(prediction_file):  # if this split has not been run yet

                if multiprocessing:
                    ray.init(_temp_dir=os.path.join(os.path.expanduser('~'), 'raytmp'))
                    best_hpams = hpam_tune_raytune(
                        model=model,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        early_stopping_dataset=(
                            early_stopping_dataset if model.early_stopping else None
                        ),
                        hpam_set=model_hpam_set,
                        response_transformation=response_transformation,
                        metric=metric,
                        ray_path=os.path.abspath(os.path.join(result_path, "raytune"))
                    )
                else:
                    best_hpams = hpam_tune(
                        model=model,
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        early_stopping_dataset=(
                            early_stopping_dataset if model.early_stopping else None
                        ),
                        hpam_set=model_hpam_set,
                        response_transformation=response_transformation,
                        metric=metric
                    )

                print(f"Best hyperparameters: {best_hpams}")
                print("Training model on full train and validation set to predict test set")
                # save best hyperparameters as json
                with open(os.path.join(predictions_path, f"best_hpams_split_{split_index}.json"), "w") as f:
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
                    response_transformation=response_transformation
                )
                test_dataset.save(prediction_file)
            else:
                print(f"Split {split_index} already exists. Skipping.")
                best_hpams = json.load(open(os.path.join(predictions_path, f"best_hpams_split_{split_index}.json")))

            if randomization_mode is not None:
                randomization_test_views = get_randomization_test_views(model=model,
                                                                        randomization_mode=randomization_mode
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
                    response_transformation=response_transformation
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
                    response_transformation=response_transformation
                )


def get_randomization_test_views(model: DRPModel, randomization_mode: List[str]) -> Dict[str, List[str]]:
    cell_line_views = model.cell_line_views
    drug_views = model.drug_views
    randomization_test_views = {}
    if "SVCC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVCC_{view}"] = [view for view in cell_line_views if view != view]
    if "SVRC" in randomization_mode:
        for view in cell_line_views:
            randomization_test_views[f"SVRC_{view}"] = [view]
    if "SVCD" in randomization_mode:
        for view in drug_views:
            randomization_test_views[f"SVCD_{view}"] = [view for view in drug_views if view != view]
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
        response_transformation: Optional[TransformerMixin] = None):
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
        trial_file = os.path.join(robustness_test_path, f"test_dataset_{test_mode}_split_{split_index}_{trial}.csv")
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
                response_transformation=response_transformation
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
        response_transformation=Optional[TransformerMixin]

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
    cl_features = model.load_cell_line_features(data_path="data", dataset_name=train_dataset.dataset_name)
    drug_features = model.load_drug_features(data_path="data", dataset_name=train_dataset.dataset_name)
    for test_name, views in randomization_test_views.items():
        randomization_test_path = os.path.join(path_out, test_name)
        randomization_test_file = os.path.join(randomization_test_path,
                                               f"test_dataset_{test_mode}_split_{split_index}.csv")

        os.makedirs(randomization_test_path, exist_ok=True)
        if not os.path.isfile(randomization_test_file):  # if this splits test has not been run yet
            for view in views:
                cl_features_rand = cl_features.copy()
                drug_features_rand = drug_features.copy()
                if view in cl_features.get_view_names():
                    cl_features_rand.randomize_features(view, randomization_type=randomization_type)
                elif view in drug_features.get_view_names():
                    drug_features_rand.randomize_features(view, randomization_type=randomization_type)
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
        print('Loading cell line features ...')
        cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    if drug_features is None:
        print('Loading drug features ...')
        drug_features = model.load_drug_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    # making sure there are no missing features:
    print('Reducing datasets ...')
    train_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    prediction_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    print('Constructing feature matrices ...')
    inputs = model.get_feature_matrices(
        cell_line_ids=train_dataset.cell_line_ids,
        drug_ids=train_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features)
    prediction_inputs = model.get_feature_matrices(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features)
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
        response_transformation.fit(train_dataset.response.reshape(-1, 1))
        train_dataset.response = response_transformation.transform(train_dataset.response.reshape(-1, 1)).squeeze()
        early_stopping_dataset.response = response_transformation.transform(
            early_stopping_dataset.response.reshape(-1, 1)).squeeze()
        prediction_dataset.response = response_transformation.transform(
            prediction_dataset.response.reshape(-1, 1)).squeeze()

    print('Training model ...')
    if model.early_stopping:
        model.train(
            output=train_dataset,
            output_earlystopping=early_stopping_dataset,
            **inputs
        )
    else:
        model.train(
            output=train_dataset,
            **inputs
        )

    prediction_dataset.predictions = model.predict(**prediction_inputs)

    if response_transformation:
        prediction_dataset.response = response_transformation.inverse_transform(prediction_dataset.response)

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
        response_transformation=response_transformation
    )
    return evaluate(validation_dataset, metric=[metric])


def hpam_tune(
        model: DRPModel,
        train_dataset: DrugResponseDataset,
        validation_dataset: DrugResponseDataset,
        hpam_set: List[Dict],
        early_stopping_dataset: Optional[DrugResponseDataset] = None,
        response_transformation: Optional[TransformerMixin] = None,
        metric: str = "rmse"
) -> Dict:
    best_score = float("inf")
    best_hyperparameters = None
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
            response_transformation=response_transformation
        )[metric]
        if score < best_score:
            print(f"current best {metric} score: {np.round(score, 3)}")
            best_score = score
            best_hyperparameters = hyperparameter
    return best_hyperparameters


def hpam_tune_raytune(
        model: DRPModel,
        train_dataset: DrugResponseDataset,
        validation_dataset: DrugResponseDataset,
        early_stopping_dataset: Optional[DrugResponseDataset],
        hpam_set: List[Dict],
        response_transformation: Optional[TransformerMixin] = None,
        metric: str = "rmse",
        ray_path: str = "raytune"
) -> Dict:
    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 1}
    else:
        resources_per_trial = {"cpu": 1}
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            path_data="data",
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation
        ),
        config=tune.grid_search(hpam_set),
        mode="min",
        num_samples=5,
        resources_per_trial=resources_per_trial,
        chdir_to_trial_dir=False,
        verbose=0,
        storage_path=ray_path
    )
    best_config = analysis.get_best_config(metric=metric, mode="min")
    return best_config

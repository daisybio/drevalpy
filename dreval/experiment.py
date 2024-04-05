from typing import Dict, List, Optional, Tuple
import warnings
from .dataset import DrugResponseDataset, FeatureDataset
from .evaluation import evaluate
from .drp_model import DRPModel
import numpy as np
import os
import shutil
from ray import tune
from sklearn.base import TransformerMixin

# TODO save hpams and their scores to disk
def drug_response_experiment(
    models: List[DRPModel],
    response_data: DrugResponseDataset,
    response_transformation: Optional[TransformerMixin] = None,
    run_id: str = "",
    test_mode: str = "LPO",
    metric: str = "rmse",
    multiprocessing: bool = False,
    randomization_mode: Optional[List[str]] = None,
    randomization_type: str = "permutation",
    path_out: str = "results/",
    overwrite: bool = False,
) -> None :
    """
    Run the drug response prediction experiment. Save results to disc.
    :param models: list of models to compare
    :param response_data: drug response dataset
    :param response_transformation: normalizer to use for the response data
    :param metric: metric to use for hyperparameter optimization
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
    :param path_out: path to the output directory
    :param run_id: identifier to save the results
    :param test_mode: test mode one of "LPO", "LCO", "LDO" (leave-pair-out, leave-cell-line-out, leave-drug-out)
    :param overwrite: whether to overwrite existing results

    :return: None
    """

    result_path = os.path.join(path_out, run_id)
    # if results exists, delete them if overwrite is true

    if os.path.exists(result_path) and overwrite:
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    # TODO load existing progress if it exists, currently we just overwrite
    for model in models:
        model_path = os.path.join(result_path, model.model_name)
        os.makedirs(model_path, exist_ok=True)
        predictions_path = os.path.join(model_path, "predictions")
        os.makedirs(predictions_path, exist_ok=True)
        if randomization_mode is not None:
            randomization_test_path = os.path.join(model_path, "randomization_tests")
            os.makedirs(randomization_test_path)

        model_hpam_set = model.get_hyperparameter_set()

        response_data.split_dataset(
            n_cv_splits=5,
            mode=test_mode,
            split_validation=True,
            validation_ratio=0.1,
            random_state=42,
        )

        for split_index, split in enumerate(response_data.cv_splits):
            train_dataset = split["train"]
            validation_dataset = split["validation"]
            test_dataset = split["test"]

            # if model.early_stopping is true then we split the validation set into a validation and early stopping set
            if model.early_stopping:
                validation_dataset, early_stopping_dataset = split_early_stopping(
                    validation_dataset=validation_dataset, test_mode=test_mode
                )

            if multiprocessing:
                best_hpams = hpam_tune_raytune(
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
            train_dataset.add_rows(
                validation_dataset
            )  # use full train val set data for final training
            train_dataset.shuffle(random_state=42)

            test_dataset = train_and_predict(
                model=model,
                hpams=best_hpams,
                train_dataset=train_dataset,
                prediction_dataset=test_dataset,
                early_stopping_dataset=(
                    early_stopping_dataset if model.early_stopping else None
                ),
                response_transformation=response_transformation
            )
            test_dataset.save(os.path.join(predictions_path, f"test_dataset_{test_mode}_split_{split_index}.csv"))

            if randomization_mode is not None:
                randomization_test_views = get_randomization_test_views(model=model,
                    randomization_mode=randomization_mode
                )
                randomization_test(
                    randomization_test_views=randomization_test_views,
                    model=model,
                    hpam_set=best_hpams,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    early_stopping_dataset=early_stopping_dataset,
                    path_out=randomization_test_path,
                    split_index=split_index,
                    test_mode=test_mode,
                    randomization_type=randomization_type,
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

def randomization_test(
    randomization_test_views: Dict[str, List[str]],
    model: DRPModel,
    hpam_set: Dict,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    path_out: str,
    split_index: int,
    test_mode: str,
    randomization_type: str = "permutation",
    response_transformation = Optional[TransformerMixin]

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
    cl_features = model.get_cell_line_features(path=hpam_set["feature_path"])
    drug_features = model.get_drug_features(path=hpam_set["feature_path"])
    for test_name, views in randomization_test_views.items():
        randomization_test_path = os.path.join(path_out, test_name)
        os.makedirs(randomization_test_path, exist_ok=True)
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
                train_dataset=train_dataset,
                prediction_dataset=test_dataset,
                early_stopping_dataset=early_stopping_dataset,
                response_transformation=response_transformation,
                cl_features=cl_features_rand,
                drug_features=drug_features_rand,
            )
            test_dataset_rand.save(os.path.join(randomization_test_path, f"test_dataset_{test_mode}_split_{split_index}.csv"))

def split_early_stopping(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> Tuple[DrugResponseDataset]:
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
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    cl_features: Optional[FeatureDataset] = None,
    drug_features: Optional[FeatureDataset] = None,
) -> DrugResponseDataset:
    if cl_features is None:
        cl_features = model.get_cell_line_features(path=hpams["feature_path"])
    if drug_features is None:
        drug_features = model.get_drug_features(path=hpams["feature_path"])
    # making sure there are no missing features:
    train_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    prediction_dataset.reduce_to(
        cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
    )

    if early_stopping_dataset is not None:
        early_stopping_dataset.reduce_to(
            cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
        )
    if response_transformation:
        response_transformation.fit(train_dataset.response.reshape(-1, 1))
        train_dataset.response = response_transformation.transform(train_dataset.response.reshape(-1, 1)).squeeze()
        early_stopping_dataset.response = response_transformation.transform(early_stopping_dataset.response.reshape(-1, 1)).squeeze()
        prediction_dataset.response = response_transformation.transform(prediction_dataset.response.reshape(-1, 1)).squeeze()


    model.train(
        cell_line_input=cl_features,
        drug_input=drug_features,
        output=train_dataset,
        hyperparameters=hpams,
        output_earlystopping=early_stopping_dataset,
    )

    prediction_dataset.predictions = model.predict(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )
    if response_transformation:
        prediction_dataset.response = response_transformation.inverse_transform(prediction_dataset.response)
    return prediction_dataset


def train_and_evaluate(
    model: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    response_transformation: Optional[TransformerMixin] = None,
    metric: str = "rmse",
) -> float:
    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
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
        score = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
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
    metric: str = "rmse"
) -> Dict:
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation
        ),
        config=tune.grid_search(hpam_set),
        mode="min",
        num_samples=len(hpam_set),
        resources_per_trial={"cpu": 1},
        chdir_to_trial_dir=False,
        verbose=0,
    )
    best_config = analysis.get_best_config(metric=metric, mode="min")
    return best_config

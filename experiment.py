from typing import Dict, List, Optional

from models_code import SimpleNeuralNetwork
from suite.data_wrapper import DrugResponseDataset, FeatureDataset
import pandas as pd
from suite.evaluation import evaluate
from suite.model_wrapper import DRPModel
from ray import tune
import numpy as np
import os


def drug_response_experiment(
    models: List[DRPModel],
    response_data: DrugResponseDataset,
    multiprocessing: bool = False,
) -> Dict[str, List[DrugResponseDataset]]:
    result = {}
    for model in models:
        result[model.model_name] = []
        model_hpam_set = model.get_hyperparameter_set()

        response_data.split_dataset(
            n_cv_splits=5,
            mode="LPO",
            split_validation=True,
            validation_ratio=0.1,
            random_state=42,
        ) 
        for split in response_data.cv_splits:
            train_dataset = split["train"]
            validation_dataset = split["validation"]
            test_dataset = split["test"]

            # if model.early_stopping is true then we split the validation set into a validation and early stopping set
            if model.early_stopping:
                validation_dataset.shuffle(random_state=42) 
                cv_v = validation_dataset.split_dataset(
                    n_cv_splits=4,
                    mode="LPO",
                    split_validation=False,
                    random_state=42,
                )
                # take the first fold of a cv as the split
                validation_dataset = cv_v[0]["train"]
                early_stopping_dataset = cv_v[0]["test"]

            if multiprocessing:
                best_hpams = hpam_tune_raytune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    early_stopping_dataset=early_stopping_dataset if model.early_stopping else None,
                    hpam_set=model_hpam_set,
                )
            else:
                best_hpams = hpam_tune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    early_stopping_dataset=early_stopping_dataset if model.early_stopping else None,
                    hpam_set=model_hpam_set,
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
                early_stopping_dataset=early_stopping_dataset if model.early_stopping else None,

            )

            result[model.model_name].append(test_dataset)
    return result


def train_and_predict(
    model: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    prediction_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
) -> DrugResponseDataset:
    cl_features = model.get_cell_line_features(path=hpams["feature_path"])
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
        
    model.train(cell_line_input=cl_features, drug_input=drug_features, output=train_dataset, hyperparameters=hpams, output_earlystopping=early_stopping_dataset)
    
    prediction_dataset.predictions = model.predict(
        cell_line_ids=prediction_dataset.cell_line_ids,
        drug_ids=prediction_dataset.drug_ids,
        cell_line_input=cl_features,
        drug_input=drug_features,
    )

    return prediction_dataset


def train_and_evaluate(
    model: DRPModel,
    hpams: Dict[str, List],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
    metric: str = "rmse",
) -> float:
    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
    )
    return evaluate(validation_dataset, metric=[metric]) 


def hpam_tune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: List[Dict],
    early_stopping_dataset: Optional[DrugResponseDataset] = None,
) -> Dict:
    best_rmse = float("inf")
    best_hyperparameters = None
    for hyperparameter in hpam_set:
        rmse = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric="rmse",
        )["rmse"]
        if rmse < best_rmse:
            print(f"current best rmse: {np.round(rmse, 3)}")
            best_rmse = rmse
            best_hyperparameters = hyperparameter
    return best_hyperparameters


def hpam_tune_raytune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: Optional[DrugResponseDataset],
    hpam_set: List[Dict],
) -> Dict:
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric="rmse",
        ),
        config=tune.grid_search(hpam_set),
        mode="min",
        num_samples=len(hpam_set),
        resources_per_trial={"cpu": 1},
        chdir_to_trial_dir=False,
        verbose=0,
    )
    best_config = analysis.get_best_config(metric="rmse", mode="min")
    return best_config


neural_net_baseline = SimpleNeuralNetwork("smpl", target="IC50")

models = [neural_net_baseline]

response_data = pd.read_csv("data/GDSC/response_GDSC2.csv")
output = response_data["LN_IC50"].values
cell_line_ids = response_data["CELL_LINE_NAME"].values
drug_ids = response_data["DRUG_NAME"].values
response_data = DrugResponseDataset(
    response=output, cell_line_ids=cell_line_ids, drug_ids=drug_ids
)
result = drug_response_experiment(models, response_data, multiprocessing=True)
print(result)

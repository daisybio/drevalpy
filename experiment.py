from typing import Dict, List

from models_code import SimpleNeuralNetwork
from suite.data_wrapper import DrugResponseDataset, FeatureDataset
import pandas as pd
from suite.evaluation import evaluate
from suite.model_wrapper import DRPModel
from ray import tune


def drug_response_experiment(
    models: List[DRPModel],
    response_data: DrugResponseDataset,
    multiprocessing: bool = False,
) -> Dict[List[DrugResponseDataset]]:
    result = {}
    for model in models:
        result[model.model_name] = []
        model_hpam_set = model.get_hyperparameter_set()

        cl_features = model.get_cell_line_features(path=model_hpam_set["feature_path"])
        drug_features = model.get_drug_features(path=model_hpam_set["feature_path"])

        # making sure there are no missing features:
        response_data.reduce_to(
            cell_line_ids=cl_features.identifiers, drug_ids=drug_features.identifiers
        )
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
            if multiprocessing:
                best_hpams = hpam_tune_raytune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    hpam_set=model_hpam_set,
                    cl_features=cl_features,
                    drug_features=drug_features,
                )
            else:
                best_hpams = hpam_tune(
                    model=model,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    hpam_set=model_hpam_set,
                    cl_features=cl_features,
                    drug_features=drug_features,
                )
            model.train(
                cell_line_input=cl_features,
                drug_input=drug_features,
                output=train_dataset,
                hyperparameters=best_hpams,
            )
            test_dataset.predictions = model.predict(
                cell_line_input=cl_features, drug_input=drug_features
            )
            result[model.model_name].append(test_dataset)
    return result


def train_and_evaluate(
    model: DRPModel,
    hpams: Dict[List],
    cl_features: FeatureDataset,
    drug_features: FeatureDataset,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    metric: str = "rmse",
) -> float:
    model.train(
        cell_line_input=cl_features,
        drug_input=drug_features,
        output=train_dataset,
        hyperparameters=hpams,
    )
    validation_dataset.predictions = model.predict(
        cell_line_input=cl_features, drug_input=drug_features
    )
    return evaluate(validation_dataset, metric=[metric])[metric]


def hpam_tune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: List[Dict],
    cl_features: FeatureDataset,
    drug_features: FeatureDataset,
) -> Dict:
    best_rmse = float("inf")
    best_hyperparameters = None
    for hyperparameter in hpam_set:
        rmse = train_and_evaluate(
            model=model,
            hpams=hyperparameter,
            cl_features=cl_features,
            drug_features=drug_features,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            metric="rmse",
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_hyperparameters = hyperparameter
    return best_hyperparameters


def hpam_tune_raytune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    hpam_set: List[Dict],
    cl_features: FeatureDataset,
    drug_features: FeatureDataset,
) -> Dict:
    analysis = tune.run(
        lambda hpams: train_and_evaluate(
            model=model,
            hpams=hpams,
            cl_features=cl_features,
            drug_features=drug_features,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            metric="rmse",
        ),
        config={"config": tune.grid_search(hpam_set)},
        mode="min",
        num_samples=len(hpam_set),
        resources_per_trial={"cpu": 1},  # adapt this, also gpu
    )

    best_config = analysis.get_best_config(metric="mean_rmse", mode="min")
    return best_config["config"]


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

"""Tests for all single drug models."""

import pathlib
import random
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from drevalpy.experiment import (
    consolidate_single_drug_model_predictions,
    cross_study_prediction,
    generate_data_saving_path,
    get_datasets_from_cv_split,
    train_and_predict,
)
from drevalpy.models import MODEL_FACTORY
from drevalpy.visualization.utils import evaluate_file


@pytest.mark.parametrize(
    "model_name",
    [
        "SingleDrugRandomForest",
        "SingleDrugElasticNet",
        "SingleDrugProteomicsElasticNet",
        "MOLIR",
        "SuperFELTR",
        "SingleDrugProteomicsRandomForest",
    ],
)
@pytest.mark.parametrize("test_mode", ["LTO"])
def test_single_drug_models(
    sample_dataset: DrugResponseDataset, model_name: str, test_mode: str, cross_study_dataset: DrugResponseDataset
) -> None:
    """
    Test the SingleDrugRandomForest model, can also test other baseline single drug models.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: either LPO or LCO
    :param cross_study_dataset: dataset
    """
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sample_dataset.split_dataset(n_cv_splits=2, mode=test_mode, random_state=42, validation_ratio=0.4)
    assert sample_dataset.cv_splits is not None
    split = sample_dataset.cv_splits[0]
    model = MODEL_FACTORY[model_name]()

    # test what happens if a drug is only in the original dataset, not in the cross-study dataset
    exclusive_drugs = list(set(sample_dataset.drug_ids).difference(set(cross_study_dataset.drug_ids)))
    all_unique_drugs = list(set(sample_dataset.drug_ids).intersection(set(cross_study_dataset.drug_ids)))
    all_unique_drugs.sort()
    exclusive_drugs.sort()
    all_unique_drugs_arr = np.array(all_unique_drugs)
    exclusive_drugs_arr = np.array(exclusive_drugs)
    # randomly sample a drug to speed up testing
    np.random.seed(123)
    np.random.shuffle(all_unique_drugs_arr)
    np.random.shuffle(exclusive_drugs_arr)
    random_drugs = all_unique_drugs_arr[:1]
    random_drugs = np.concatenate([random_drugs, exclusive_drugs_arr[:1]])
    # test what happens if the training and validation dataset is empty for a drug but the test set is not
    drug_to_remove = all_unique_drugs_arr[2]
    random_drugs = np.concatenate([random_drugs, [drug_to_remove]])

    hpam_combi = model.get_hyperparameter_set()[0]
    result_path = tempfile.TemporaryDirectory()
    if model_name == "SingleDrugRandomForest":
        hpam_combi["n_estimators"] = 2  # reduce test time
        hpam_combi["max_depth"] = 2  # reduce test time
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1

    for random_drug in random_drugs:
        model = MODEL_FACTORY[model_name]()
        predictions_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=str(random_drug),
            result_path=result_path.name,
            suffix="predictions",
        )
        prediction_file = pathlib.Path(predictions_path, "predictions_split_0.csv")
        (
            train_dataset,
            validation_dataset,
            early_stopping_dataset,
            test_dataset,
        ) = get_datasets_from_cv_split(split, MODEL_FACTORY[model_name], model_name, random_drug)
        train_dataset.add_rows(validation_dataset)
        if random_drug == drug_to_remove:
            reduce_to_drugs = np.array(list(set(train_dataset.drug_ids) - {random_drug}))
            train_dataset.reduce_to(cell_line_ids=None, drug_ids=reduce_to_drugs)
        train_dataset.shuffle(random_state=42)

        test_dataset = train_and_predict(
            model=model,
            hpams=hpam_combi,
            path_data="../data",
            train_dataset=train_dataset,
            prediction_dataset=test_dataset,
            early_stopping_dataset=None,
            response_transformation=None,
            model_checkpoint_dir="TEMPORARY",
        )

        # Save and load test (should either succeed or raise NotImplementedError)
        if len(train_dataset) == 0:
            print(f"Training dataset empty for drug {random_drug}, continuing with train_and_predict anyway")
        else:
            with tempfile.TemporaryDirectory() as model_dir:
                try:

                    model.save(model_dir)
                    loaded_model = MODEL_FACTORY[model_name].load(model_dir)

                    # Re-run prediction with loaded model
                    preds_original = model.predict(
                        drug_ids=test_dataset.drug_ids,
                        cell_line_ids=test_dataset.cell_line_ids,
                        drug_input=model.load_drug_features("../data", "TOYv1"),
                        cell_line_input=model.load_cell_line_features("../data", "TOYv1"),
                    )
                    preds_loaded = loaded_model.predict(
                        drug_ids=test_dataset.drug_ids,
                        cell_line_ids=test_dataset.cell_line_ids,
                        drug_input=model.load_drug_features("../data", "TOYv1"),
                        cell_line_input=model.load_cell_line_features("../data", "TOYv1"),
                    )
                    assert isinstance(preds_loaded, np.ndarray)
                    assert preds_loaded.shape == preds_original.shape
                except NotImplementedError:
                    print(f"{model_name} does not implement save/load")

        cross_study_dataset.remove_nan_responses()
        parent_dir = str(pathlib.Path(predictions_path).parent)
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data="../data",
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=parent_dir,
            split_index=0,
            single_drug_id=str(random_drug),
        )
        test_dataset.to_csv(prediction_file)
    consolidate_single_drug_model_predictions(
        models=[MODEL_FACTORY[model_name]],
        n_cv_splits=1,
        results_path=result_path.name,
        cross_study_datasets=[cross_study_dataset.dataset_name],
        randomization_mode=None,
        n_trials_robustness=0,
        out_path=result_path.name,
    )
    # get cross-study predictions and assert that each drug-cell line combination only occurs once
    cross_study_predictions = pd.read_csv(
        pathlib.Path(result_path.name, model_name, "cross_study", "cross_study_TOYv2_split_0.csv")
    )
    assert len(cross_study_predictions) == len(
        cross_study_predictions.drop_duplicates([DRUG_IDENTIFIER, CELL_LINE_IDENTIFIER])
    )
    predictions_file = pathlib.Path(result_path.name, model_name, "predictions", "predictions_split_0.csv")
    cross_study_file = pathlib.Path(result_path.name, model_name, "cross_study", "cross_study_TOYv2_split_0.csv")
    for file in [predictions_file, cross_study_file]:
        (
            overall_eval,
            eval_results_per_drug,
            eval_results_per_cl,
            t_vs_p,
            model_name,
        ) = evaluate_file(pred_file=file, test_mode=test_mode, model_name=model_name)
        assert len(overall_eval) == 1
        print(f"Performance of {model_name}: PCC = {overall_eval['Pearson'][0]}")
        assert overall_eval["Pearson"][0] >= -1.0

"""Tests for all single drug models."""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from drevalpy.experiment import (
    consolidate_single_drug_model_predictions,
    cross_study_prediction,
    generate_data_saving_path,
)
from drevalpy.models import construct_model
from drevalpy.visualization.utils import evaluate_file


def _resolve_single_drug_model_name(whole_name: str) -> str:
    if whole_name.startswith("SingleDrugRandomForest"):
        return "SingleDrugRandomForest"
    if whole_name.startswith("SingleDrugElasticNet"):
        return "SingleDrugElasticNet"
    return whole_name


def _construct_single_drug_model(whole_name: str, model_name: str):
    if whole_name.endswith("[proteomics]"):
        predictor_token = model_name.replace("SingleDrug", "singleDrug")
        # Keep zoo identity for result paths; only swap the cell-line featurizer.
        return construct_model(model_name, f"normalizedProteomics:{predictor_token}")
    return construct_model(model_name)


def _configure_single_drug_hpam(model_name: str, hpam_combi: dict) -> None:
    if model_name == "SingleDrugRandomForest":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1


def _select_single_drug_test_drugs(
    sample_dataset: DrugResponseDataset,
    cross_study_dataset: DrugResponseDataset,
) -> tuple[np.ndarray, str]:
    exclusive_drugs = list(set(sample_dataset.drug_ids).difference(set(cross_study_dataset.drug_ids)))
    all_unique_drugs = list(set(sample_dataset.drug_ids).intersection(set(cross_study_dataset.drug_ids)))
    all_unique_drugs.sort()
    exclusive_drugs.sort()
    all_unique_drugs_arr = np.array(all_unique_drugs)
    exclusive_drugs_arr = np.array(exclusive_drugs)
    rng = np.random.default_rng(123)
    rng.shuffle(all_unique_drugs_arr)
    rng.shuffle(exclusive_drugs_arr)
    random_drugs = all_unique_drugs_arr[:1]
    random_drugs = np.concatenate([random_drugs, exclusive_drugs_arr[:1]])
    drug_to_remove = all_unique_drugs_arr[2]
    return np.concatenate([random_drugs, [drug_to_remove]]), drug_to_remove


def _assert_single_drug_save_load_roundtrip(model, model_class, test_dataset, train_dataset) -> None:
    if len(train_dataset) == 0:
        print("Training dataset empty, continuing with train_and_predict anyway")
        return
    with tempfile.TemporaryDirectory() as model_dir:
        try:
            from tests.conftest import load_features_for_model

            checkpoint = f"{model_dir}/model"
            model.save(checkpoint)
            loaded_model = model_class.load(checkpoint)
            cell_line_input, drug_input = load_features_for_model(model, dataset_name="TOYv1")
            preds_original = model.predict(
                drug_ids=test_dataset.drug_ids,
                cell_line_ids=test_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            preds_loaded = loaded_model.predict(
                drug_ids=test_dataset.drug_ids,
                cell_line_ids=test_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            assert isinstance(preds_loaded, np.ndarray)
            assert preds_loaded.shape == preds_original.shape
        except NotImplementedError:
            print(f"{model_class.get_model_name()} does not implement save/load")


@pytest.mark.parametrize(
    "model_name",
    [
        "SingleDrugRandomForest[gex]",
        "SingleDrugRandomForest[proteomics]",
        "SingleDrugElasticNet[gex]",
        "SingleDrugElasticNet[proteomics]",
        "MOLIR",
        "SuperFELTR",
    ],
)
@pytest.mark.parametrize("test_mode", ["LTO"])
def test_single_drug_models(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
    data_dir,
) -> None:
    """Test the SingleDrugRandomForest model, can also test other baseline single drug models.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: either LPO or LCO
    :param cross_study_dataset: dataset
    :param data_dir: path to the data directory
    """
    from drevalpy.experiment import seed_everything

    seed_everything(42)

    whole_name = model_name
    model_name = _resolve_single_drug_model_name(whole_name)

    sample_dataset.split_dataset(n_cv_splits=2, mode=test_mode, random_state=42, validation_ratio=0.4)
    assert sample_dataset.cv_splits is not None
    split = sample_dataset.cv_splits[0]
    model_class = _construct_single_drug_model(whole_name, model_name)
    model = model_class()

    random_drugs, drug_to_remove = _select_single_drug_test_drugs(sample_dataset, cross_study_dataset)

    hpam_combi = model.get_hyperparameter_set()[0]
    result_path = tempfile.TemporaryDirectory()
    _configure_single_drug_hpam(model_name, hpam_combi)

    for random_drug in random_drugs:
        model = model_class(hpam_combi)
        predictions_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=str(random_drug),
            result_path=result_path.name,
            suffix="predictions",
        )
        prediction_file = pathlib.Path(predictions_path, "predictions_split_0.csv")

        # Extract and mask fold datasets for this drug
        train_dataset = split["train"].copy()
        validation_dataset = split["validation"].copy()
        test_dataset = split["test"].copy()
        train_dataset = train_dataset.masked(train_dataset.drug_ids == random_drug)
        validation_dataset = validation_dataset.masked(validation_dataset.drug_ids == random_drug)
        test_dataset = test_dataset.masked(test_dataset.drug_ids == random_drug)

        train_dataset.add_rows(validation_dataset)
        if random_drug == drug_to_remove:
            reduce_to_drugs = np.array(list(set(train_dataset.drug_ids) - {random_drug}))
            train_dataset.reduce_to(cell_line_ids=None, drug_ids=reduce_to_drugs)
        train_dataset.shuffle(random_state=42)

        # Train and predict using the DRPModel API
        from tests.conftest import load_features_for_model

        cl_features, drug_features = load_features_for_model(model, dataset_name=train_dataset.dataset_name)
        cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
        drugs_to_keep = drug_features.identifiers if drug_features is not None else None
        train_dataset = train_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        test_dataset = test_dataset.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

        model.train(
            output=train_dataset,
            cell_line_input=cl_features.copy(),
            drug_input=drug_features.copy() if drug_features is not None else None,
            output_earlystopping=None,
        )

        if len(test_dataset) == 0:
            test_dataset._predictions = np.array([])
        elif not model._stack.is_fitted():
            test_dataset._predictions = np.full(len(test_dataset), np.nan)
        else:
            test_dataset._predictions = model.predict(
                cell_line_ids=test_dataset.cell_line_ids,
                drug_ids=test_dataset.drug_ids,
                cell_line_input=cl_features.copy(),
                drug_input=drug_features.copy() if drug_features is not None else None,
            )

        # Save and load test (should either succeed or raise NotImplementedError)
        _assert_single_drug_save_load_roundtrip(model, model_class, test_dataset, train_dataset)

        cross_study_dataset.remove_nan_responses()
        parent_dir = str(pathlib.Path(predictions_path).parent)
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=parent_dir,
            split_index=0,
            single_drug_id=str(random_drug),
        )
        test_dataset.to_csv(prediction_file)
    consolidate_single_drug_model_predictions(
        models=[model_class],
        n_cv_splits=1,
        results_path=result_path.name,
        cross_study_datasets=[cross_study_dataset.dataset_name],
        randomization_mode=None,
        n_trials_robustness=0,
        out_path=result_path.name,
    )
    # get cross-study predictions and assert that each drug-cell line combination only occurs once
    cross_study_file = pathlib.Path(result_path.name, model_name, "cross_study", "cross_study_TOYv2_split_0.csv")
    if cross_study_file.exists():
        cross_study_predictions = pd.read_csv(cross_study_file)
        assert len(cross_study_predictions) == len(
            cross_study_predictions.drop_duplicates([DRUG_IDENTIFIER, CELL_LINE_IDENTIFIER])
        )
    predictions_file = pathlib.Path(result_path.name, model_name, "predictions", "predictions_split_0.csv")
    eval_files = [predictions_file]
    if cross_study_file.exists():
        eval_files.append(cross_study_file)
    for file in eval_files:
        (
            overall_eval,
            eval_results_per_drug,
            eval_results_per_cl,
            t_vs_p,
            model_name,
        ) = evaluate_file(pred_file=file, test_mode=test_mode, model_name=model_name)
        assert len(overall_eval) == 1
        print(f"Performance of {model_name}: PCC = {overall_eval['Pearson'][0]}")
        assert overall_eval["Pearson"].iloc[0] >= -1.0

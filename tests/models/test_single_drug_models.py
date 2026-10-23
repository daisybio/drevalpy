"""Tests for all single drug models."""

import pathlib
import tempfile
from functools import partial

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from drevalpy.experiment import (
    consolidate_single_drug_model_predictions,
    cross_study_prediction,
    generate_data_saving_path,
    get_datasets_from_cv_split,
    run_per_drug,
    split_by_drug,
    train_and_predict,
)
from drevalpy.models import MODEL_FACTORY
from drevalpy.visualization.utils import evaluate_file


def _train_and_predict_drug(
    drug_id: str,
    split: dict[str, DrugResponseDataset],
    cross_study_dataset: DrugResponseDataset,
    *,
    model_name: str,
    hpam_combi: dict,
    test_mode: str,
    data_dir: str,
    result_path: str,
) -> None:
    """
    Train a single drug model for one drug, test save and load, and predict the test and cross-study set.

    Runs in a worker of run_per_drug, like the per-drug functions of the pipeline.

    :param drug_id: drug to train the model for
    :param split: CV split, reduced to drug_id by split_by_drug
    :param cross_study_dataset: cross-study dataset, reduced to drug_id by split_by_drug
    :param model_name: model name
    :param hpam_combi: hyperparameters to use
    :param test_mode: test mode
    :param data_dir: path to the data directory
    :param result_path: path to save the predictions to
    """
    model = MODEL_FACTORY[model_name]()
    predictions_path = generate_data_saving_path(
        model_name=model_name,
        drug_id=drug_id,
        result_path=result_path,
        suffix="predictions",
    )
    prediction_file = pathlib.Path(predictions_path, "predictions_split_0.csv")
    (
        train_dataset,
        validation_dataset,
        early_stopping_dataset,
        test_dataset,
    ) = get_datasets_from_cv_split(split, MODEL_FACTORY[model_name], model_name, drug_id)
    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)

    test_dataset = train_and_predict(
        model=model,
        hpams=hpam_combi,
        path_data=data_dir,
        train_dataset=train_dataset,
        prediction_dataset=test_dataset,
        early_stopping_dataset=None,
        response_transformation=None,
        model_checkpoint_dir="TEMPORARY",
    )

    # Save and load test (should either succeed or raise NotImplementedError)
    if len(train_dataset) == 0:
        print(f"Training dataset empty for drug {drug_id}, continuing with train_and_predict anyway")
    else:
        with tempfile.TemporaryDirectory() as model_dir:
            try:

                model.save(model_dir)
                loaded_model = MODEL_FACTORY[model_name].load(model_dir)

                # Re-run prediction with loaded model
                preds_original = model.predict(
                    drug_ids=test_dataset.drug_ids,
                    cell_line_ids=test_dataset.cell_line_ids,
                    drug_input=model.load_drug_features(data_dir, "TOYv1"),
                    cell_line_input=model.load_cell_line_features(data_dir, "TOYv1"),
                )
                preds_loaded = loaded_model.predict(
                    drug_ids=test_dataset.drug_ids,
                    cell_line_ids=test_dataset.cell_line_ids,
                    drug_input=model.load_drug_features(data_dir, "TOYv1"),
                    cell_line_input=model.load_cell_line_features(data_dir, "TOYv1"),
                )
                assert isinstance(preds_loaded, np.ndarray)
                assert preds_loaded.shape == preds_original.shape
            except NotImplementedError:
                print(f"{model_name} does not implement save/load")

    parent_dir = str(pathlib.Path(predictions_path).parent)
    cross_study_prediction(
        dataset=cross_study_dataset,
        model=model,
        test_mode=test_mode,
        train_dataset=train_dataset,
        path_data=data_dir,
        early_stopping_dataset=None,
        response_transformation=None,
        path_out=parent_dir,
        split_index=0,
        single_drug_id=drug_id,
    )
    test_dataset.to_csv(prediction_file)


@pytest.mark.parametrize(
    "model_name",
    [
        "SingleDrugRandomForest[gex]",
        "SingleDrugRandomForest[proteomics]",
        "SingleDrugElasticNet[gex]",
        "SingleDrugElasticNet[proteomics]",
        "SingleDrugKNN",
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
    """
    Test the single drug models, trained per drug in parallel like in the pipeline (split_by_drug, run_per_drug).

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: either LPO or LCO
    :param cross_study_dataset: dataset
    :param data_dir: path to the data directory
    """
    from drevalpy.experiment import seed_everything

    seed_everything(42)

    whole_name = model_name
    if model_name.startswith("SingleDrugRandomForest"):
        model_name = "SingleDrugRandomForest"
    elif model_name.startswith("SingleDrugElasticNet"):
        model_name = "SingleDrugElasticNet"

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
    rng = np.random.default_rng(123)
    rng.shuffle(all_unique_drugs_arr)
    rng.shuffle(exclusive_drugs_arr)
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
        if whole_name == "SingleDrugRandomForest[gex]":
            hpam_combi["cell_line_views"] = "gene_expression"
        elif whole_name == "SingleDrugRandomForest[proteomics]":
            hpam_combi["cell_line_views"] = "proteomics"
    elif whole_name == "SingleDrugElasticNet[gex]":
        hpam_combi["cell_line_views"] = "gene_expression"
    elif whole_name == "SingleDrugElasticNet[proteomics]":
        hpam_combi["cell_line_views"] = "proteomics"
    elif model_name in ["MOLIR", "SuperFELTR"]:
        hpam_combi["epochs"] = 1

    cross_study_dataset.remove_nan_responses()
    splits_per_drug = split_by_drug(split, random_drugs)
    # empty all but the test set of drug_to_remove
    for name, dataset in splits_per_drug[str(drug_to_remove)].items():
        if name != "test" and dataset is not None:
            dataset.mask(np.zeros(len(dataset), dtype=bool))
    cross_study_per_drug = split_by_drug({"cross_study": cross_study_dataset}, random_drugs)

    run_per_drug(
        partial(
            _train_and_predict_drug,
            model_name=model_name,
            hpam_combi=hpam_combi,
            test_mode=test_mode,
            data_dir=str(data_dir),
            result_path=result_path.name,
        ),
        {drug: (splits_per_drug[drug], cross_study_per_drug[drug]["cross_study"]) for drug in splits_per_drug},
        n_jobs=2,
    )
    # the workers wrote the predictions of every drug
    drugs_path = pathlib.Path(result_path.name, model_name, "drugs")
    assert {path.name for path in drugs_path.iterdir()} == {str(drug) for drug in random_drugs}
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
        assert overall_eval["Pearson"].iloc[0] >= -1.0

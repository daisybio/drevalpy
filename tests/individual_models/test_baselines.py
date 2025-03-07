"""Tests for the baselines in the models module."""

import pathlib
import tempfile
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, Ridge

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import evaluate
from drevalpy.experiment import (
    consolidate_single_drug_model_predictions,
    cross_study_prediction,
    generate_data_saving_path,
    get_datasets_from_cv_split,
    train_and_predict,
)
from drevalpy.models import (
    MODEL_FACTORY,
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaivePredictor,
)
from drevalpy.models.baselines.sklearn_models import SklearnModel
from drevalpy.models.drp_model import DRPModel
from drevalpy.visualization.utils import evaluate_file


@pytest.mark.parametrize(
    "model_name",
    [
        "NaivePredictor",
        "NaiveDrugMeanPredictor",
        "NaiveCellLineMeanPredictor",
        "NaiveMeanEffectsPredictor",
        "ElasticNet",
        "RandomForest",
        "SVR",
        "MultiOmicsRandomForest",
        "GradientBoosting",
    ],
)
@pytest.mark.parametrize("test_mode", ["LPO", "LCO", "LDO"])
def test_baselines(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
) -> None:
    """
    Test the baselines.

    :param sample_dataset: from conftest.py
    :param model_name: name of the model
    :param test_mode: either LPO, LCO, or LDO
    :param cross_study_dataset: dataset
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode=test_mode,
        validation_ratio=0.2,
    )
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]

    model = MODEL_FACTORY[model_name]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")

    if drug_input is None:
        raise ValueError("Drug input is None")

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_pred_before = len(val_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val dataset from {len_pred_before} to {len(val_dataset)}")

    if model_name == "NaivePredictor":
        model = _call_naive_predictor(train_dataset, val_dataset, cell_line_input, test_mode)
    elif model_name == "NaiveDrugMeanPredictor":
        model = _call_naive_group_predictor(
            "drug",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    elif model_name == "NaiveCellLineMeanPredictor":
        model = _call_naive_group_predictor(
            "cell_line",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    elif model_name == "NaiveMeanEffectsPredictor":
        model = _call_naive_mean_effects_predictor(train_dataset, val_dataset, cell_line_input, drug_input, test_mode)
    else:
        model = _call_other_baselines(
            model_name,
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
        )
    # make temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running cross-study prediction for {model_name}")
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data="../data",
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "SingleDrugRandomForest",
        "SingleDrugElasticNet",
        "SingleDrugProteomicsElasticNet",
    ],
)
@pytest.mark.parametrize("test_mode", ["LPO", "LCO"])
def test_single_drug_baselines(
    sample_dataset: DrugResponseDataset, model_name: str, test_mode: str, cross_study_dataset: DrugResponseDataset
) -> None:
    """
    Test the SingleDrugRandomForest model, can also test other baseline single drug models.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: either LPO or LCO
    :param cross_study_dataset: dataset
    """
    sample_dataset.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
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
    for random_drug in random_drugs:
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
            train_dataset.reduce_to(cell_line_ids=None, drug_ids=list(set(train_dataset.drug_ids) - {random_drug}))
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
        models=[model],
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
    assert len(cross_study_predictions) == len(cross_study_predictions.drop_duplicates(["drug_id", "cell_line_id"]))
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


def _call_naive_predictor(
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    test_mode: str,
) -> DRPModel:
    """
    Call the NaivePredictor model.

    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param cell_line_input: features cell lines
    :param test_mode: either LPO, LCO, or LDO
    :returns: NaivePredictor model
    """
    naive = NaivePredictor()
    naive.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=None)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids, drug_ids=val_dataset.drug_ids, cell_line_input=cell_line_input
    )
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean
    assert np.all(val_dataset.predictions == train_mean)
    metrics = evaluate(val_dataset, metric=["Pearson"])
    assert metrics["Pearson"] == 0.0
    print(f"{test_mode}: Performance of NaivePredictor: PCC = {metrics['Pearson']}")
    return naive


def _assert_group_mean(
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    group_ids: dict[str, np.ndarray],
    naive_means: dict[int, float],
) -> None:
    """
    Assert the group mean.

    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param group_ids: group ids
    :param naive_means: means
    """
    common_ids = np.intersect1d(group_ids["train"], group_ids["val"])
    random_id = np.random.choice(common_ids)
    group_mean = train_dataset.response[group_ids["train"] == random_id].mean()
    assert group_mean == naive_means[random_id]
    assert val_dataset.predictions is not None
    assert np.all(val_dataset.predictions[group_ids["val"] == random_id] == group_mean)


def _call_naive_group_predictor(
    group: str,
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    drug_input: FeatureDataset,
    test_mode: str,
) -> DRPModel:
    naive: NaiveDrugMeanPredictor | NaiveCellLineMeanPredictor
    if group == "drug":
        naive = NaiveDrugMeanPredictor()
    else:
        naive = NaiveCellLineMeanPredictor()
    naive.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids, drug_ids=val_dataset.drug_ids, cell_line_input=cell_line_input
    )
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean
    if (group == "drug" and test_mode == "LDO") or (group == "cell_line" and test_mode == "LCO"):
        assert np.all(val_dataset.predictions == train_mean)
    elif group == "drug":
        assert isinstance(naive, NaiveDrugMeanPredictor)
        _assert_group_mean(
            train_dataset,
            val_dataset,
            group_ids={
                "train": train_dataset.drug_ids,
                "val": val_dataset.drug_ids,
            },
            naive_means=naive.drug_means,
        )
    else:  # group == "cell_line"
        assert isinstance(naive, NaiveCellLineMeanPredictor)
        _assert_group_mean(
            train_dataset,
            val_dataset,
            group_ids={
                "train": train_dataset.cell_line_ids,
                "val": val_dataset.cell_line_ids,
            },
            naive_means=naive.cell_line_means,
        )
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of {naive.get_model_name()}: PCC = {metrics['Pearson']}")
    if (group == "drug" and test_mode == "LDO") or (group == "cell_line" and test_mode == "LCO"):
        assert metrics["Pearson"] == 0.0
    return naive


def _call_other_baselines(
    model: str,
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    drug_input: FeatureDataset,
) -> DRPModel:
    """
    Call the other baselines.

    :param model: model name
    :param train_dataset: training
    :param val_dataset: validation
    :param cell_line_input: features cell lines
    :param drug_input: features drugs
    :returns: model instance
    """
    model_class = cast(type[DRPModel], MODEL_FACTORY[model])
    hpams = model_class.get_hyperparameter_set()
    if len(hpams) > 2:
        hpams = hpams[:2]
    model_instance = model_class()
    assert isinstance(model_instance, SklearnModel)
    for hpam_combi in hpams:
        if model == "RandomForest" or model == "GradientBoosting":
            hpam_combi["n_estimators"] = 2
            hpam_combi["max_depth"] = 2
            if model == "GradientBoosting":
                hpam_combi["subsample"] = 0.1
        elif model == "MultiOmicsRandomForest":
            hpam_combi["n_components"] = 10
        model_instance.build_model(hpam_combi)
        if model == "ElasticNet":
            if hpam_combi["l1_ratio"] == 0.0:
                assert issubclass(type(model_instance.model), Ridge)
            else:
                assert issubclass(type(model_instance.model), ElasticNet)

        # smaller dataset for faster testing
        train_dataset.remove_rows(indices=np.array([list(range(len(train_dataset) - 1000))]))
        model_instance.train(
            output=train_dataset,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        val_dataset._predictions = model_instance.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert val_dataset.predictions is not None
        metrics = evaluate(val_dataset, metric=["Pearson"])
        assert metrics["Pearson"] >= -1
    return model_instance


def _call_naive_mean_effects_predictor(
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    drug_input: FeatureDataset,
    test_mode: str,
) -> DRPModel:
    """
    Test the NaiveMeanEffectsPredictor model.

    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param cell_line_input: features cell lines
    :param drug_input: features drugs
    :param test_mode: either LPO, LCO, or LDO
    :returns: NaiveMeanEffectsPredictor model
    """
    naive = NaiveMeanEffectsPredictor()
    naive.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids,
        drug_ids=val_dataset.drug_ids,
        cell_line_input=cell_line_input,
    )

    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean

    # Check that predictions are within a reasonable range
    assert np.all(np.isfinite(val_dataset.predictions))
    assert np.all(val_dataset.predictions >= np.min(train_dataset.response))
    assert np.all(val_dataset.predictions <= np.max(train_dataset.response))

    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of NaiveMeanEffectsPredictor: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1  # Should be within valid Pearson range
    return naive

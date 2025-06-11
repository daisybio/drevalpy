"""Tests for the baselines in the models module that are not single drug models."""

import tempfile
from typing import cast

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Ridge

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import (
    MODEL_FACTORY,
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaivePredictor,
    NaiveTissueMeanPredictor,
)
from drevalpy.models.baselines.sklearn_models import SklearnModel
from drevalpy.models.drp_model import DRPModel


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
        "ProteomicsRandomForest",
        "ProteomicsElasticNet",
    ],
)
@pytest.mark.parametrize("test_mode", ["LTO", "LPO", "LCO", "LDO"])
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
    :param test_mode: either LPO, LCO, LDO, or LTO
    :param cross_study_dataset: dataset
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode=test_mode,
        validation_ratio=0.4,
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
    elif model_name == "NaiveTissueMeanPredictor":
        model = _call_naive_group_predictor(
            "tissue",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    else:
        model = _call_other_baselines(
            model_name,
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
        )
    # Save and load test
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        loaded_model = MODEL_FACTORY[model_name].load(model_dir)

        preds_before = model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        preds_after = loaded_model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert isinstance(preds_after, np.ndarray)
        assert preds_after.shape == preds_before.shape

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
    assert len(common_ids) > 0, (
        f"No common group identifiers found between training and validation. "
        f"Train IDs: {np.unique(group_ids['train'])}, "
        f"Val IDs: {np.unique(group_ids['val'])}"
    )
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
    naive: NaiveDrugMeanPredictor | NaiveCellLineMeanPredictor | NaiveTissueMeanPredictor
    if group == "drug":
        naive = NaiveDrugMeanPredictor()
    elif group == "cell_line":
        naive = NaiveCellLineMeanPredictor()
    elif group == "tissue":
        naive = NaiveTissueMeanPredictor()
    else:
        raise ValueError(f"Unknown group: {group}")
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
    if (
        (group == "drug" and test_mode == "LDO")
        or (group == "cell_line" and test_mode in ["LCO", "LTO"])
        or (group == "tissue" and test_mode == "LTO")
    ):
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
    elif group == "cell_line":
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
    elif group == "tissue":
        assert isinstance(naive, NaiveTissueMeanPredictor)
        if train_dataset.tissue is None or val_dataset.tissue is None:
            raise ValueError("Tissue information is missing in the dataset.")
        _assert_group_mean(
            train_dataset,
            val_dataset,
            group_ids={
                "train": train_dataset.tissue,
                "val": val_dataset.tissue,
            },
            naive_means=naive.tissue_means,
        )
    else:
        raise ValueError(f"Unknown group: {group}")
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
        print(metrics)
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
    assert np.all(
        val_dataset.predictions >= 2 * np.min(train_dataset.response) - 1e-6
    ), f"Predictions below min response: {np.min(val_dataset.predictions)} < {np.min(train_dataset.response)}"
    assert np.all(val_dataset.predictions <= 2 * np.max(train_dataset.response) + 1e-6), (
        f"Predictions above max response: {np.max(val_dataset.predictions)} > {np.max(train_dataset.response)},"
        f"Problematic cell line: {val_dataset.cell_line_ids[np.argmax(val_dataset.predictions)]}, "
        f"Problematic drug: {val_dataset.drug_ids[np.argmax(val_dataset.predictions)]},"
        f"CL effect: {naive.cell_line_effects[val_dataset.cell_line_ids[np.argmax(val_dataset.predictions)]]}, "
        f"Drug effect: {naive.drug_effects[val_dataset.drug_ids[np.argmax(val_dataset.predictions)]]}, "
        f"Dataset mean: {naive.dataset_mean}"
    )

    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of NaiveMeanEffectsPredictor: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1  # Should be within valid Pearson range
    return naive

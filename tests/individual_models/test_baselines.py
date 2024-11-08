"""Tests for the baselines in the models module."""
import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Ridge

from drevalpy.evaluation import evaluate, pearson
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import (
    MODEL_FACTORY,
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaivePredictor,
    SingleDrugRandomForest,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "NaivePredictor",
        "NaiveDrugMeanPredictor",
        "NaiveCellLineMeanPredictor",
        "ElasticNet",
        "RandomForest",
        "SVR",
        "MultiOmicsRandomForest",
        "GradientBoosting",
    ],
)
@pytest.mark.parametrize("test_mode", ["LPO", "LCO", "LDO"])
def test_baselines(
        sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset],
        model_name: str,
        test_mode: str
) -> None:
    """
    Test the baselines.

    :param sample_dataset: from conftest.py
    :param model_name: name of the model
    :param test_mode: either LPO, LCO, or LDO
    """
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_pred_before = len(val_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val dataset from {len_pred_before} to {len(val_dataset)}")

    if model_name == "NaivePredictor":
        _call_naive_predictor(train_dataset, val_dataset, test_mode)
    elif model_name == "NaiveDrugMeanPredictor":
        _call_naive_group_predictor(
            "drug",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    elif model_name == "NaiveCellLineMeanPredictor":
        _call_naive_group_predictor(
            "cell_line",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    else:
        _call_other_baselines(
            model_name,
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
        )


@pytest.mark.parametrize("model_name", ["SingleDrugRandomForest"])
@pytest.mark.parametrize("test_mode", ["LPO", "LCO"])
def test_single_drug_baselines(
        sample_dataset: tuple[DrugResponseDataset, FeatureDataset, FeatureDataset],
        model_name: str,
        test_mode: str
) -> None:
    """
    Test the SingleDrugRandomForest model, can also test other baseline single drug models.

    :param sample_dataset: from conftest.py
    :param model_name: model name
    :param test_mode: either LPO or LCO
    """
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]

    all_predictions = np.zeros_like(val_dataset.drug_ids, dtype=float)
    for drug in np.unique(train_dataset.drug_ids):
        model = SingleDrugRandomForest()
        hpam_combi = model.get_hyperparameter_set()[0]
        hpam_combi["n_estimators"] = 2  # reduce test time
        hpam_combi["max_depth"] = 2  # reduce test time
        model.build_model(hpam_combi)
        output_mask = train_dataset.drug_ids == drug
        drug_train = train_dataset.copy()
        drug_train.mask(output_mask)
        model.train(output=drug_train, cell_line_input=cell_line_input)

        val_mask = val_dataset.drug_ids == drug
        all_predictions[val_mask] = model.predict(
            drug_ids=drug,
            cell_line_ids=val_dataset.cell_line_ids[val_mask],
            cell_line_input=cell_line_input,
        )
        pcc_drug = pearson(val_dataset.response[val_mask], all_predictions[val_mask])
        print(f"{test_mode}: Performance of {model_name} for drug {drug}: PCC = {pcc_drug}")
    val_dataset.predictions = all_predictions
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Collapsed performance of {model_name}: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] > 0.0


def _call_naive_predictor(
        train_dataset: DrugResponseDataset,
        val_dataset: DrugResponseDataset,
        test_mode: str
) -> None:
    """
    Call the NaivePredictor model.

    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param test_mode: either LPO, LCO, or LDO
    """
    naive = NaivePredictor()
    naive.train(output=train_dataset)
    val_dataset.predictions = naive.predict(cell_line_ids=val_dataset.cell_line_ids)
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean
    assert np.all(val_dataset.predictions == train_mean)
    metrics = evaluate(val_dataset, metric=["Pearson"])
    assert metrics["Pearson"] == 0.0
    print(f"{test_mode}: Performance of NaivePredictor: PCC = {metrics['Pearson']}")


def _assert_group_mean(
        train_dataset: DrugResponseDataset,
        val_dataset: DrugResponseDataset,
        group_ids: dict[str, np.ndarray],
        naive_means: dict[int, float]
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
    assert np.all(val_dataset.predictions[group_ids["val"] == random_id] == group_mean)


def _call_naive_group_predictor(
        group: str,
        train_dataset: DrugResponseDataset,
        val_dataset: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        test_mode: str
) -> None:
    if group == "drug":
        naive = NaiveDrugMeanPredictor()
    else:
        naive = NaiveCellLineMeanPredictor()
    naive.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    val_dataset.predictions = naive.predict(cell_line_ids=val_dataset.cell_line_ids, drug_ids=val_dataset.drug_ids)
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean
    if (group == "drug" and test_mode == "LDO") or (group == "cell_line" and test_mode == "LCO"):
        assert np.all(val_dataset.predictions == train_mean)
    elif group == "drug":
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
    print(f"{test_mode}: Performance of {naive.model_name}: PCC = {metrics['Pearson']}")
    if (group == "drug" and test_mode == "LDO") or (group == "cell_line" and test_mode == "LCO"):
        assert metrics["Pearson"] == 0.0


def _call_other_baselines(
        model: str,
        train_dataset: DrugResponseDataset,
        val_dataset: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
) -> None:
    """
    Call the other baselines.

    :param model: model name
    :param train_dataset: training
    :param val_dataset: validation
    :param cell_line_input: features cell lines
    :param drug_input: features drugs
    """
    model_class = MODEL_FACTORY[model]
    hpams = model_class.get_hyperparameter_set()
    if len(hpams) > 2:
        hpams = hpams[:2]
    model_instance = model_class()
    for hpam_combi in hpams:
        if model == "RandomForest" or model == "GradientBoosting":
            hpam_combi["n_estimators"] = 2
            hpam_combi["max_depth"] = 2
            if model == "GradientBoosting":
                hpam_combi["subsample"] = 0.1
        model_instance.build_model(hpam_combi)
        if model == "ElasticNet":
            if hpam_combi["l1_ratio"] == 0.0:
                assert issubclass(type(model_instance.model), Ridge)
            else:
                assert issubclass(type(model_instance.model), ElasticNet)
        train_dataset.remove_rows(indices=np.array([list(range(len(train_dataset) - 1000))]))  # smaller dataset for
        # faster testing
        model_instance.train(
            output=train_dataset,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        val_dataset.predictions = model_instance.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert val_dataset.predictions is not None
        metrics = evaluate(val_dataset, metric=["Pearson"])
        assert metrics["Pearson"] >= -1

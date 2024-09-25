import pytest
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet

from drevalpy.evaluation import evaluate, pearson
from drevalpy.models import (
    NaivePredictor,
    NaiveDrugMeanPredictor,
    NaiveCellLineMeanPredictor,
    SingleDrugRandomForest,
    MODEL_FACTORY,
)
from .utils import sample_dataset, call_save_and_load


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
def test_baselines(sample_dataset, model_name, test_mode):
    drug_response, cell_line_input, drug_input = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=5,
        mode=test_mode,
    )
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]
    if model_name == "NaivePredictor":
        call_naive_predictor(train_dataset, val_dataset, test_mode)
    elif model_name == "NaiveDrugMeanPredictor":
        call_naive_group_predictor(
            "drug", train_dataset, val_dataset, cell_line_input, drug_input, test_mode
        )
    elif model_name == "NaiveCellLineMeanPredictor":
        call_naive_group_predictor(
            "cell_line",
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )
    else:
        call_other_baselines(
            model_name,
            train_dataset,
            val_dataset,
            cell_line_input,
            drug_input,
            test_mode,
        )


@pytest.mark.parametrize("model_name", ["SingleDrugRandomForest"])
@pytest.mark.parametrize("test_mode", ["LPO", "LCO"])
def test_single_drug_baselines(sample_dataset, model_name, test_mode):
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
        print(
            f"{test_mode}: Performance of {model_name} for drug {drug}: PCC = {pcc_drug}"
        )
    val_dataset.predictions = all_predictions
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(
        f"{test_mode}: Collapsed performance of {model_name}: PCC = {metrics['Pearson']}"
    )
    assert metrics["Pearson"] > 0.0


def call_naive_predictor(train_dataset, val_dataset, test_mode):
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
    call_save_and_load(naive)


def assert_group_mean(train_dataset, val_dataset, group_ids, naive_means):
    common_ids = np.intersect1d(group_ids["train"], group_ids["val"])
    random_id = np.random.choice(common_ids)
    group_mean = train_dataset.response[group_ids["train"] == random_id].mean()
    assert group_mean == naive_means[random_id]
    assert np.all(val_dataset.predictions[group_ids["val"] == random_id] == group_mean)


def call_naive_group_predictor(
    group, train_dataset, val_dataset, cell_line_input, drug_input, test_mode
):
    if group == "drug":
        naive = NaiveDrugMeanPredictor()
    else:
        naive = NaiveCellLineMeanPredictor()
    naive.train(
        output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input
    )
    val_dataset.predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids, drug_ids=val_dataset.drug_ids
    )
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert train_mean == naive.dataset_mean
    if (group == "drug" and test_mode == "LDO") or (
        group == "cell_line" and test_mode == "LCO"
    ):
        assert np.all(val_dataset.predictions == train_mean)
    elif group == "drug":
        assert_group_mean(
            train_dataset,
            val_dataset,
            group_ids={"train": train_dataset.drug_ids, "val": val_dataset.drug_ids},
            naive_means=naive.drug_means,
        )
    else:  # group == "cell_line"
        assert_group_mean(
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
    if (group == "drug" and test_mode == "LDO") or (
        group == "cell_line" and test_mode == "LCO"
    ):
        assert metrics["Pearson"] == 0.0
    call_save_and_load(naive)


def call_other_baselines(
    model, train_dataset, val_dataset, cell_line_input, drug_input, test_mode
):
    model_class = MODEL_FACTORY[model]
    hpams = model_class.get_hyperparameter_set()
    if len(hpams) > 3:
        hpams = hpams[:3]
    model_instance = model_class()
    for hpam_combi in hpams:
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
        val_dataset.predictions = model_instance.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert val_dataset.predictions is not None
        metrics = evaluate(val_dataset, metric=["Pearson"])
        print(
            f"{test_mode}: Performance of {model}, hpams: {hpam_combi}: PCC = {metrics['Pearson']}"
        )
        if model == "ElasticNet" and hpam_combi["l1_ratio"] == 1.0:
            # TODO: Why is this happening? Investigate
            assert metrics["Pearson"] == 0.0
        elif model == "ElasticNet" and hpam_combi["l1_ratio"] == 0.5:
            # TODO: Why so bad? E.g., LPO+l1_ratio=0.5 -> 0.06, LCO+l1_ratio=0.5 -> 0.1, LDO+l1_ratio=0.5 -> 0.23
            assert metrics["Pearson"] > 0.0
        elif test_mode == "LDO":
            assert metrics["Pearson"] > 0.0
        else:
            assert metrics["Pearson"] > 0.5
    call_save_and_load(model_instance)

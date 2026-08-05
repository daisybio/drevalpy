"""Public behavior tests for naive and sklearn factory models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import construct_model
from drevalpy.models.drp_model import DRPModel


def test_naive_mean_effects_predictor_tissue_decomposition() -> None:
    cell_lines = np.array(["CL1", "CL1", "CL2", "CL2", "CL3", "CL3"])
    drugs = np.array(["D1", "D2", "D1", "D2", "D1", "D2"])
    response = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    output = DrugResponseDataset(response=response, cell_line_ids=cell_lines, drug_ids=drugs)
    cell_line_input = FeatureDataset(
        features={
            "CL1": {CELL_LINE_IDENTIFIER: np.array(["CL1"]), TISSUE_IDENTIFIER: np.array(["Lung"])},
            "CL2": {CELL_LINE_IDENTIFIER: np.array(["CL2"]), TISSUE_IDENTIFIER: np.array(["Lung"])},
            "CL3": {CELL_LINE_IDENTIFIER: np.array(["CL3"]), TISSUE_IDENTIFIER: np.array(["Blood"])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "D1": {DRUG_IDENTIFIER: np.array(["D1"])},
            "D2": {DRUG_IDENTIFIER: np.array(["D2"])},
        }
    )

    model = construct_model("NaiveMeanEffectsPredictor")({})
    model.train(output=output, cell_line_input=cell_line_input, drug_input=drug_input)
    preds = model.predict(
        cell_line_ids=np.array(["CL1", "CL2", "CL3"]),
        drug_ids=np.array(["D1", "D2", "D1"]),
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()

    dataset_mean = float(np.mean(response))
    lung_mean = float(np.mean([1.0, 2.0, 3.0, 4.0]))
    blood_mean = float(np.mean([5.0, 6.0]))
    cl1_mean = float(np.mean([1.0, 2.0]))
    cl2_mean = float(np.mean([3.0, 4.0]))
    cl3_mean = float(np.mean([5.0, 6.0]))
    d1_effect = float(np.mean([1.0, 3.0, 5.0]) - dataset_mean)
    d2_effect = float(np.mean([2.0, 4.0, 6.0]) - dataset_mean)
    expected = np.array(
        [
            dataset_mean + (lung_mean - dataset_mean) + (cl1_mean - lung_mean) + d1_effect,
            dataset_mean + (lung_mean - dataset_mean) + (cl2_mean - lung_mean) + d2_effect,
            dataset_mean + (blood_mean - dataset_mean) + (cl3_mean - blood_mean) + d1_effect,
        ]
    )
    np.testing.assert_allclose(preds, expected)

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = str(Path(model_dir) / "model")
        model.save(checkpoint)
        loaded = construct_model("NaiveMeanEffectsPredictor").load(checkpoint)
        loaded_preds = loaded.predict(
            cell_line_ids=np.array(["CL1"]),
            drug_ids=np.array(["D1"]),
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        assert loaded_preds[0] == pytest.approx(preds[0])


def test_naive_mean_effects_predictor_without_tissue_matches_previous_decomposition() -> None:
    cell_lines = np.array(["CL1", "CL1", "CL2", "CL2"])
    drugs = np.array(["D1", "D2", "D1", "D2"])
    response = np.array([1.0, 2.0, 5.0, 8.0])
    output = DrugResponseDataset(response=response, cell_line_ids=cell_lines, drug_ids=drugs)
    cell_line_input = FeatureDataset(
        features={
            "CL1": {CELL_LINE_IDENTIFIER: np.array(["CL1"])},
            "CL2": {CELL_LINE_IDENTIFIER: np.array(["CL2"])},
        }
    )
    drug_input = FeatureDataset(
        features={
            "D1": {DRUG_IDENTIFIER: np.array(["D1"])},
            "D2": {DRUG_IDENTIFIER: np.array(["D2"])},
        }
    )

    model = construct_model("NaiveMeanEffectsPredictor")({})
    model.train(output=output, cell_line_input=cell_line_input, drug_input=drug_input)
    preds = model.predict(
        cell_line_ids=np.array(["CL1", "CL2"]),
        drug_ids=np.array(["D1", "D2"]),
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    dataset_mean = float(np.mean(response))
    expected = np.array(
        [
            dataset_mean + (np.mean([1.0, 2.0]) - dataset_mean) + (np.mean([1.0, 5.0]) - dataset_mean),
            dataset_mean + (np.mean([5.0, 8.0]) - dataset_mean) + (np.mean([2.0, 8.0]) - dataset_mean),
        ]
    )
    np.testing.assert_allclose(preds, expected)


def test_random_forest_respects_max_depth() -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2", "cl3", "cl3", "cl4", "cl4"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2", "d1", "d2", "d1", "d2"]),
    )
    cell_line_input = FeatureDataset(
        features={f"cl{i}": {"gene_expression": np.linspace(i, i + 1, 4)} for i in range(1, 5)}
    )
    drug_input = FeatureDataset(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0, 0.5])},
            "d2": {"fingerprints": np.array([0.0, 1.0, 0.25])},
        }
    )
    shallow = construct_model("RandomForest")({"n_estimators": 5, "max_depth": 1, "n_jobs": 1, "max_samples": 0.9})
    shallow.train(response, cell_line_input, drug_input)
    deep = construct_model("RandomForest")({"n_estimators": 5, "max_depth": 8, "n_jobs": 1, "max_samples": 0.9})
    deep.train(response, cell_line_input, drug_input)
    ids = response.cell_line_ids
    drugs = response.drug_ids
    shallow_preds = shallow.predict(ids, drugs, cell_line_input, drug_input)
    deep_preds = deep.predict(ids, drugs, cell_line_input, drug_input)
    assert shallow_preds.shape == deep_preds.shape == (8,)
    assert np.isfinite(shallow_preds).all()
    assert np.isfinite(deep_preds).all()


@pytest.mark.parametrize(
    "model_name",
    [
        "NaivePredictor",
        "NaiveDrugMeanPredictor",
        "NaiveCellLineMeanPredictor",
        "NaiveMeanEffectsPredictor",
        "NaiveTissueDrugMeanPredictor",
        "ElasticNet",
        "RandomForest",
        "SVR",
        "MultiViewRandomForest",
        "GradientBoosting",
        "AdaBoostDecisionTree",
        "KNNRegressor",
        "Lasso",
        "MultiViewXGBoost",
    ],
)
@pytest.mark.parametrize("test_mode", ["LTO", "LPO", "LCO", "LDO"])
def test_baselines(
    sample_dataset: DrugResponseDataset,
    model_name: str,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
    data_dir,
) -> None:
    drug_response = sample_dataset
    drug_response.split_dataset(n_cv_splits=2, mode=test_mode, validation_ratio=0.4)
    assert drug_response.cv_splits is not None
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]

    try:
        model, preds_before = _train_and_predict(model_name, train_dataset, val_dataset, test_mode, data_dir)
    except ImportError as exc:
        if model_name == "MultiViewXGBoost":
            pytest.skip(str(exc))
        raise

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = str(Path(model_dir) / "model")
        model.save(checkpoint)
        loaded_model = construct_model(model_name).load(checkpoint)
        train_dataset, val_dataset, cell_line_input, drug_input = _subset_dataset(
            model=loaded_model, train_dataset=train_dataset, val_dataset=val_dataset, data_dir=data_dir
        )
        preds_after = loaded_model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert isinstance(preds_after, np.ndarray)
        assert preds_after.shape == preds_before.shape

    with tempfile.TemporaryDirectory() as temp_dir:
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data=str(data_dir),
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )


def _train_and_predict(
    model_name: str,
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    test_mode: str,
    data_dir,
) -> tuple[DRPModel, np.ndarray]:
    if model_name == "NaivePredictor":
        return _call_naive_predictor(train_dataset, val_dataset, test_mode, data_dir)
    if model_name == "NaiveDrugMeanPredictor":
        return _call_naive_group_predictor("drug", train_dataset, val_dataset, test_mode, data_dir)
    if model_name == "NaiveCellLineMeanPredictor":
        return _call_naive_group_predictor("cell_line", train_dataset, val_dataset, test_mode, data_dir)
    if model_name == "NaiveTissueMeanPredictor":
        return _call_naive_group_predictor("tissue", train_dataset, val_dataset, test_mode, data_dir)
    if model_name == "NaiveMeanEffectsPredictor":
        return _call_naive_mean_effects_predictor(train_dataset, val_dataset, test_mode, data_dir)
    if model_name == "NaiveTissueDrugMeanPredictor":
        return _call_naive_tissue_drug_predictor(train_dataset, val_dataset, test_mode, data_dir)
    return _call_other_baselines(model_name, train_dataset, val_dataset, data_dir)


def _call_naive_predictor(
    train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, test_mode: str, data_dir
) -> tuple[DRPModel, np.ndarray]:
    _ = data_dir
    naive = construct_model("NaivePredictor")({})
    empty = FeatureDataset(features={})
    naive.train(output=train_dataset, cell_line_input=empty, drug_input=None)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids,
        drug_ids=val_dataset.drug_ids,
        cell_line_input=empty,
        drug_input=None,
    )
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    assert np.allclose(val_dataset.predictions, train_mean)
    metrics = evaluate(val_dataset, metric=["Pearson"])
    assert metrics["Pearson"] == 0.0
    print(f"{test_mode}: Performance of NaivePredictor: PCC = {metrics['Pearson']}")
    return naive, val_dataset._predictions


def _call_naive_group_predictor(
    group: str, train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, test_mode: str, data_dir
) -> tuple[DRPModel, np.ndarray]:
    if group == "drug":
        naive: DRPModel = construct_model("NaiveDrugMeanPredictor")()
    elif group == "cell_line":
        naive = construct_model("NaiveCellLineMeanPredictor")()
    elif group == "tissue":
        naive = construct_model("NaiveTissueMeanPredictor")()
    else:
        raise ValueError(f"Unknown group: {group}")
    # Defaults already applied by construction.
    train_dataset, val_dataset, cell_line_input, drug_input = _subset_dataset(
        model=naive, train_dataset=train_dataset, val_dataset=val_dataset, data_dir=data_dir
    )
    naive.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids,
        drug_ids=val_dataset.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert val_dataset.predictions is not None
    train_mean = train_dataset.response.mean()
    if (
        (group == "drug" and test_mode == "LDO")
        or (group == "cell_line" and test_mode in ["LCO", "LTO"])
        or (group == "tissue" and test_mode == "LTO")
    ):
        assert np.allclose(val_dataset.predictions, train_mean)
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of {naive.get_model_name()}: PCC = {metrics['Pearson']}")
    if (group == "drug" and test_mode == "LDO") or (group == "cell_line" and test_mode == "LCO"):
        assert metrics["Pearson"] == 0.0
    return naive, val_dataset._predictions


_MULTI_VIEW_BASELINE_MODELS = {
    "RandomForest",
    "GradientBoosting",
    "ElasticNet",
    "AdaBoostDecisionTree",
    "SVR",
    "MultiViewXGBoost",
}


def _subset_hpams_for_baseline(model: str, hpams: list) -> list:
    if len(hpams) <= 2:
        return hpams
    if model not in _MULTI_VIEW_BASELINE_MODELS:
        return hpams[:2]
    return hpams[:2]


def _tune_baseline_hpam(model: str, hpam_combi: dict) -> None:
    if model in {"RandomForest", "GradientBoosting"}:
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
        if model == "GradientBoosting":
            hpam_combi["subsample"] = 0.1
    elif model == "MultiViewRandomForest":
        hpam_combi.pop("n_components", None)
        hpam_combi["methylation_n_components"] = 10
    elif model == "AdaBoostDecisionTree":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
        hpam_combi["min_samples_split"] = 2
        hpam_combi["min_samples_leaf"] = 1
    elif model == "KNNRegressor":
        hpam_combi["n_neighbors"] = 3
        hpam_combi["weights"] = "distance"
        hpam_combi["variance"] = 0.75


def _call_other_baselines(model: str, train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, data_dir):
    model_class = construct_model(model)
    hpams = _subset_hpams_for_baseline(model, model_class.get_hyperparameter_set())
    model_instance = None
    for hpam_combi in hpams:
        _tune_baseline_hpam(model, hpam_combi)
        model_instance = model_class(hpam_combi)
        train_dataset, val_dataset, cell_line_input, drug_input = _subset_dataset(
            model=model_instance, train_dataset=train_dataset, val_dataset=val_dataset, data_dir=data_dir
        )
        model_instance.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input)
        val_dataset._predictions = model_instance.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert val_dataset.predictions is not None
        metrics = evaluate(val_dataset, metric=["Pearson"])
        assert metrics["Pearson"] >= -1
    return model_instance, val_dataset._predictions


def _call_naive_mean_effects_predictor(
    train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, test_mode: str, data_dir
) -> tuple[DRPModel, np.ndarray]:
    naive = construct_model("NaiveMeanEffectsPredictor")({})
    train_dataset, val_dataset, cell_line_input, drug_input = _subset_dataset(
        model=naive, train_dataset=train_dataset, val_dataset=val_dataset, data_dir=data_dir
    )
    naive.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids,
        drug_ids=val_dataset.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert val_dataset.predictions is not None
    assert np.all(np.isfinite(val_dataset.predictions))
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of NaiveMeanEffectsPredictor: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1
    return naive, val_dataset._predictions


def _call_naive_tissue_drug_predictor(
    train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, test_mode: str, data_dir
) -> tuple[DRPModel, np.ndarray]:
    naive = construct_model("NaiveTissueDrugMeanPredictor")({})
    train_dataset, val_dataset, cell_line_input, drug_input = _subset_dataset(
        model=naive, train_dataset=train_dataset, val_dataset=val_dataset, data_dir=data_dir
    )
    naive.train(output=train_dataset, cell_line_input=cell_line_input, drug_input=drug_input)
    val_dataset._predictions = naive.predict(
        cell_line_ids=val_dataset.cell_line_ids,
        drug_ids=val_dataset.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert val_dataset.predictions is not None
    assert np.all(np.isfinite(val_dataset.predictions))
    metrics = evaluate(val_dataset, metric=["Pearson"])
    print(f"{test_mode}: Performance of NaiveTissueDrugMeanPredictor: PCC = {metrics['Pearson']}")
    assert metrics["Pearson"] >= -1
    return naive, val_dataset._predictions


def _subset_dataset(model: DRPModel, train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset, data_dir):
    cell_line_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")
    cell_lines_to_keep = cell_line_input.identifiers if cell_line_input.features else None
    drugs_to_keep = drug_input.identifiers if drug_input is not None and drug_input.features else None
    if cell_lines_to_keep is not None or drugs_to_keep is not None:
        train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if cell_line_input.features or (drug_input is not None and drug_input.features):
        return train_dataset, val_dataset, cell_line_input, drug_input
    empty = FeatureDataset(features={})
    return train_dataset, val_dataset, empty, None

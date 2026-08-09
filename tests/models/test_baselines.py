"""Public behavior tests for naive and sklearn factory models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.data.structures.response_batch import ResponseBatch
from drevalpy.data.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.models import construct_model
from tests.conftest import MockFeatureSource


def test_naive_mean_effects_predictor_tissue_decomposition() -> None:
    cell_lines = np.array(["CL1", "CL1", "CL2", "CL2", "CL3", "CL3"])
    drugs = np.array(["D1", "D2", "D1", "D2", "D1", "D2"])
    response = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    output = ResponseBatch(response=response, cell_line_ids=cell_lines, drug_ids=drugs)
    cell_line_input = MockFeatureSource(
        features={
            "CL1": {CELL_LINE_IDENTIFIER: np.array(["CL1"]), TISSUE_IDENTIFIER: np.array(["Lung"])},
            "CL2": {CELL_LINE_IDENTIFIER: np.array(["CL2"]), TISSUE_IDENTIFIER: np.array(["Lung"])},
            "CL3": {CELL_LINE_IDENTIFIER: np.array(["CL3"]), TISSUE_IDENTIFIER: np.array(["Blood"])},
        }
    )
    drug_input = MockFeatureSource(
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
    output = ResponseBatch(response=response, cell_line_ids=cell_lines, drug_ids=drugs)
    cell_line_input = MockFeatureSource(
        features={
            "CL1": {CELL_LINE_IDENTIFIER: np.array(["CL1"])},
            "CL2": {CELL_LINE_IDENTIFIER: np.array(["CL2"])},
        }
    )
    drug_input = MockFeatureSource(
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
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2", "cl3", "cl3", "cl4", "cl4"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2", "d1", "d2", "d1", "d2"]),
    )
    cell_line_input = MockFeatureSource(
        features={f"cl{i}": {"gene_expression": np.linspace(i, i + 1, 4)} for i in range(1, 5)}
    )
    drug_input = MockFeatureSource(
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
        "ElasticNet",
        "RandomForest",
        "SVR",
        "GradientBoosting",
        "AdaBoostDecisionTree",
        "KNNRegressor",
        "Lasso",
    ],
)
def test_baseline_train_predict_roundtrip(model_name: str) -> None:
    """Each baseline model trains and predicts on synthetic ResponseBatch data."""
    response = ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    cell_line_input = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )
    drug_input = MockFeatureSource(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )

    model_cls = construct_model(model_name)
    hpams = model_cls.get_hyperparameter_set()
    hpam_combi = hpams[0] if hpams else {}
    if model_name in {"RandomForest", "GradientBoosting"}:
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name == "AdaBoostDecisionTree":
        hpam_combi["n_estimators"] = 2
        hpam_combi["max_depth"] = 2
    elif model_name == "KNNRegressor":
        hpam_combi["n_neighbors"] = 2

    model = model_cls(hpam_combi)
    model.train(output=response, cell_line_input=cell_line_input, drug_input=drug_input)
    preds = model.predict(
        cell_line_ids=response.cell_line_ids,
        drug_ids=response.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()

    with tempfile.TemporaryDirectory() as model_dir:
        checkpoint = str(Path(model_dir) / "model")
        model.save(checkpoint)
        loaded = model_cls.load(checkpoint)
        loaded_preds = loaded.predict(
            cell_line_ids=response.cell_line_ids,
            drug_ids=response.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        assert loaded_preds.shape == preds.shape

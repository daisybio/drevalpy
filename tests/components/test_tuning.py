"""Tests for Ray/Optuna tuning helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.search_space import dict_to_ray_space, merge_model_config_spaces
from drevalpy.models import construct_model
from drevalpy.models.config import model_config_from_spec


def test_dict_to_ray_space_converts_structured_specs() -> None:
    pytest.importorskip("ray")
    space = dict_to_ray_space(
        {
            "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
            "predictor.randomForest.max_samples": {
                "type": "float",
                "low": 0.1,
                "high": 0.9,
                "default": 0.2,
            },
        }
    )
    assert "predictor.randomForest.n_estimators" in space
    assert "predictor.randomForest.max_samples" in space


def test_construct_model_merged_space_has_indexed_concat_keys() -> None:
    import drevalpy.components.register_builtins as register_builtins

    register_builtins.register_builtin_components()
    model_cls = construct_model("ComboRF", "pca[expression]+landmarkGenes:fingerprints:randomForest")
    config = model_config_from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    merged = merge_model_config_spaces(config)
    assert any("cell_line_featurizer.pca[expression]." in key for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    assert model_cls.get_structured_hyperparameter_space() == merged


def test_hpam_tune_uses_optuna(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    from drevalpy.components.tuning import hpam_tune

    captured: dict[str, object] = {}

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            captured["param_space"] = param_space
            captured["search_alg"] = tune_config.search_alg
            captured["num_samples"] = tune_config.num_samples

        def fit(self):
            class Result:
                config = {"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = {"RMSE": 0.1}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

            return Results()

    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)
    monkeypatch.setattr("ray.init", lambda **kwargs: None)

    from drevalpy.datasets.dataset import DrugResponseDataset

    model_cls = construct_model("ElasticNet")
    train = DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1"]),
    )
    val = train.copy()
    best = hpam_tune(
        model_class=model_cls,
        train_dataset=train,
        validation_dataset=val,
        early_stopping_dataset=None,
        metric="RMSE",
        path_data="data",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=3),
    )
    assert "alpha" in best
    assert captured["num_samples"] == 3
    assert captured["search_alg"] is not None


@pytest.mark.skipif(os.environ.get("DREVALPY_RUN_RAY_TESTS") != "1", reason="optional Ray runtime test")
def test_hpam_tune_smoke(tmp_path, data_dir) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    import numpy as np

    from drevalpy import experiment
    from drevalpy.datasets.dataset import DrugResponseDataset

    model_cls = construct_model("ElasticNet")
    model = model_cls()
    cell_line_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")
    valid_cell_lines = list(cell_line_input.identifiers)[:2]
    valid_drugs = list(drug_input.identifiers)[:2]
    responses = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    cell_line_ids = np.array([valid_cell_lines[0], valid_cell_lines[0], valid_cell_lines[1], valid_cell_lines[1]])
    drug_ids = np.array([valid_drugs[0], valid_drugs[1], valid_drugs[0], valid_drugs[1]])
    train_dataset = DrugResponseDataset(
        response=responses,
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        dataset_name="TOYv1",
    )
    val_dataset = train_dataset.copy()
    train_dataset.reduce_to(cell_line_ids=cell_line_input.identifiers, drug_ids=drug_input.identifiers)
    val_dataset.reduce_to(cell_line_ids=cell_line_input.identifiers, drug_ids=drug_input.identifiers)

    best = experiment.hpam_tune(
        model_class=model_cls,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        early_stopping_dataset=None,
        metric="RMSE",
        path_data=str(data_dir),
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2, storage_path=str(tmp_path)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

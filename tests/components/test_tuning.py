"""Tests for Ray/Optuna tuning helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.search_space import dict_to_ray_space, merge_model_config_spaces
from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig


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
    config = from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any("cell_line_featurizer.pca[expression]." in key for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    assert model_cls.get_structured_hyperparameter_space() == merged


def test_hpam_tune_uses_optuna(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    from drevalpy.components.tuning.hpo import mu_hpam_tune
    from drevalpy.data.splitting import EntityScope

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

    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    model_cls = construct_model("ElasticNet")
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    train_scope = EntityScope(cell_lines=np.array([0, 1]), drugs=np.array([0, 1]))
    val_scope = EntityScope(cell_lines=np.array([0, 1]), drugs=np.array([0, 1]))
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=3),
    )
    assert "alpha" in best
    assert captured["num_samples"] == 3
    assert captured["search_alg"] is not None


@pytest.mark.skipif(os.environ.get("DREVALPY_RUN_RAY_TESTS") != "1", reason="optional Ray runtime test")
def test_hpam_tune_smoke(tmp_path, data_dir) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    from drevalpy import experiment
    from drevalpy.data import load_mudataset
    from drevalpy.data.splitting import MuDataSplitter
    from drevalpy.experiment.fold import prepare_mu_fold

    model_cls = construct_model("ElasticNet")
    mudataset = load_mudataset("TOYv1")
    splitter = MuDataSplitter()
    folds = splitter.split(mudataset, mode="LPO", n_splits=2, validation_ratio=0.4)
    split = folds[0]
    fold_data = prepare_mu_fold(mudataset, split, model_cls)

    best = experiment.mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=fold_data.train_scope,
        val_scope=fold_data.val_scope,
        early_stopping_scope=fold_data.early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2, storage_path=str(tmp_path)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

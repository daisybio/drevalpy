"""Tests for Optuna tuning helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import optuna
import pytest

from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.search_space import merge_model_config_spaces, sample_from_optuna_trial


def test_sample_from_optuna_trial_converts_structured_specs() -> None:
    space = {
        "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
        "predictor.randomForest.max_samples": {
            "type": "float",
            "low": 0.1,
            "high": 0.9,
            "default": 0.2,
        },
    }
    study = optuna.create_study()
    trial = study.ask()
    sampled = sample_from_optuna_trial(trial, space)
    assert "predictor.randomForest.n_estimators" in sampled
    assert "predictor.randomForest.max_samples" in sampled
    assert 10 <= sampled["predictor.randomForest.n_estimators"] <= 20
    assert 0.1 <= sampled["predictor.randomForest.max_samples"] <= 0.9


def test_sample_from_optuna_trial_log_scale() -> None:
    space = {
        "alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True, "default": 1.0},
    }
    study = optuna.create_study()
    trial = study.ask()
    sampled = sample_from_optuna_trial(trial, space)
    assert 0.001 <= sampled["alpha"] <= 10.0


def test_sample_from_optuna_trial_categorical() -> None:
    space = {
        "kernel": {"type": "categorical", "choices": ["linear", "rbf", "poly"], "default": "rbf"},
    }
    study = optuna.create_study()
    trial = study.ask()
    sampled = sample_from_optuna_trial(trial, space)
    assert sampled["kernel"] in ["linear", "rbf", "poly"]


def test_sample_from_optuna_trial_passthrough_non_mapping() -> None:
    space = {"fixed_param": 42}
    study = optuna.create_study()
    trial = study.ask()
    sampled = sample_from_optuna_trial(trial, space)
    assert sampled["fixed_param"] == 42


def test_construct_model_merged_space_has_indexed_concat_keys() -> None:
    from drevalpy.registry._builtins import register_builtin_components

    register_builtin_components()
    model_cls = construct_model("ComboRF", "pca[expression]+landmarkGenes:fingerprints:randomForest")
    config = from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any("cell_line_featurizer.pca[expression]." in key for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    assert model_cls.get_structured_hyperparameter_space() == merged


@patch(
    "drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics",
    return_value=({"RMSE": 0.1}, np.zeros(4)),
)
def test_hpam_tune_uses_optuna(mock_evaluate) -> None:
    from drevalpy.models.tuning.hpo import hpam_tune
    from drevalpy.types import SplitMask
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    model_cls = construct_model("ElasticNet")
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    shape = mudataset.response_matrix.shape
    train_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    val_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=3),
    )
    assert "alpha" in best
    assert mock_evaluate.call_count == 3


@pytest.mark.skipif(os.environ.get("DREVALPY_RUN_HPO_TESTS") != "1", reason="optional HPO runtime test")
def test_hpam_tune_smoke(synthetic_dataset) -> None:
    from drevalpy.models.tuning.hpo import hpam_tune
    from drevalpy.registry.splitter import get as get_splitter

    model_cls = construct_model("ElasticNet")
    splitter = get_splitter("LPO")
    folds = splitter(synthetic_dataset, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    early_stopping_scope = None
    val_scope = split.val
    if model_cls.supports_early_stopping() and len(split.val) > 1:
        early_stopping_scope, val_scope = split.early_stopping_mask()

    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=synthetic_dataset,
        train_scope=split.train,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

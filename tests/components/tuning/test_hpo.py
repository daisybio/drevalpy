"""Deterministic mocked tests for Optuna HPO."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from drevalpy.components.core.tuning.config import HPOConfig
from drevalpy.components.core.tuning.hpo import mu_hpam_tune
from drevalpy.data.structures import EntityScope
from drevalpy.models import construct_model
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def _tiny_mudataset_and_scopes():
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    train_scope = EntityScope(pairs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    val_scope = EntityScope(pairs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    return mudataset, train_scope, val_scope


def test_hpam_tune_no_space_returns_defaults(monkeypatch) -> None:
    model_cls = construct_model("ElasticNet")
    monkeypatch.setattr(
        "drevalpy.components.core.tuning.hpo.structured_space_for_drp_model",
        lambda _cls: {},
    )
    monkeypatch.setattr(
        "drevalpy.components.core.tuning.hpo.has_tunable_hyperparameters",
        lambda _cls: False,
    )

    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=5),
    )
    assert best == model_cls.get_default_hyperparameters()


def test_hpam_tune_zero_trials_returns_defaults() -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=0),
    )
    assert best == model_cls.get_default_hyperparameters()


@patch("drevalpy.components.core.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.1)
def test_hpam_tune_one_trial(mock_evaluate) -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
    mock_evaluate.assert_called_once()


@patch("drevalpy.components.core.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=float("nan"))
def test_hpam_tune_all_nan_returns_defaults(mock_evaluate) -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert best == model_cls.get_default_hyperparameters()


@patch("drevalpy.components.core.tuning.hpo_runtime._mu_evaluate_trial_model", side_effect=RuntimeError("boom"))
def test_hpam_tune_trial_exception_returns_defaults(mock_evaluate) -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert best == model_cls.get_default_hyperparameters()


def test_hpam_tune_rejects_metric_mismatch() -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    with pytest.raises(ValueError, match="must match"):
        mu_hpam_tune(
            model_class=model_cls,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=None,
            metric="Pearson",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
        )


@patch("drevalpy.components.core.tuning.hpo_runtime._mu_evaluate_trial_model")
def test_hpam_tune_multiple_trials_picks_best(mock_evaluate) -> None:
    scores = iter([0.8, 0.3, 0.5])
    mock_evaluate.side_effect = lambda *args, **kwargs: next(scores)

    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best = mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=3),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

"""Deterministic mocked tests for Optuna HPO."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.models import construct_model
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo import HPOTrialsFailedError, hpam_tune
from drevalpy.types import SplitMask
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def _tiny_mudataset_and_scopes():
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    shape = mudataset.response_matrix.shape
    train_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    val_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    return mudataset, train_scope, val_scope


def test_hpam_tune_no_space_returns_defaults(monkeypatch) -> None:
    model_cls = construct_model("ElasticNet")
    monkeypatch.setattr(model_cls, "get_structured_hyperparameter_space", classmethod(lambda cls: {}))
    monkeypatch.setattr(
        "drevalpy.models.tuning.hpo.has_tunable_hyperparameters",
        lambda _cls: False,
    )

    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best, _ = hpam_tune(
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
    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=0),
    )
    assert best == model_cls.get_default_hyperparameters()


@patch(
    "drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics",
    return_value=({"RMSE": 0.1}, np.zeros(4)),
)
@pytest.mark.parametrize("n_trials", [1, 3])
def test_hpam_tune_evaluates_exactly_n_trials(mock_evaluate, n_trials) -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=n_trials),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
    assert mock_evaluate.call_count == n_trials


@patch(
    "drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics",
    return_value=({"RMSE": float("nan")}, np.zeros(4)),
)
def test_hpam_tune_all_nan_returns_defaults(mock_evaluate) -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert best == model_cls.get_default_hyperparameters()


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics", side_effect=RuntimeError("boom"))
def test_hpam_tune_raises_when_every_trial_fails(mock_evaluate) -> None:
    """Every trial raised, so the study produced no tuning information.

    Reporting defaults as a result would hide the real cause behind a later traceback.
    """
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    with pytest.raises(HPOTrialsFailedError, match="All 2 hyperparameter trials failed") as excinfo:
        hpam_tune(
            model_class=model_cls,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        )

    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "boom"


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics")
def test_hpam_tune_warns_when_some_trials_fail(mock_evaluate, caplog) -> None:
    """A partly failing study still tuned, so it warns and returns the survivors' best."""
    calls = {"n": 0}

    def evaluate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"RMSE": 0.1}, np.zeros(4)

    mock_evaluate.side_effect = evaluate

    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    with caplog.at_level(logging.WARNING):
        best, _ = hpam_tune(
            model_class=model_cls,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        )

    assert isinstance(best, dict)
    assert "alpha" in best
    assert "1 of 2 hyperparameter trials failed" in caplog.text


def test_hpam_tune_rejects_metric_mismatch() -> None:
    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    with pytest.raises(ValueError, match="must match"):
        hpam_tune(
            model_class=model_cls,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=None,
            metric="Pearson",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
        )


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics")
def test_hpam_tune_multiple_trials_picks_best(mock_evaluate) -> None:
    scores = iter([0.8, 0.3, 0.5])
    mock_evaluate.side_effect = lambda *args, **kwargs: ({"RMSE": next(scores)}, np.zeros(4))

    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    best, _ = hpam_tune(
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


class TestUnmockedTuning:
    """The one case that runs Optuna and the estimator for real, end to end.

    Everything above mocks the trial evaluation, so nothing above would notice a
    break between ``hpam_tune`` and a real split, a real fit and a real metric.
    Kept in a class of its own so the ``slow`` marker covers only this test and
    not the mocked ones, which are milliseconds each.
    """

    pytestmark = pytest.mark.slow

    def test_tuning_elastic_net_over_a_real_lpo_fold_returns_its_hyperparameters(self, synthetic_dataset) -> None:
        from drevalpy.registry.splitter import get as get_splitter

        model_cls = construct_model("ElasticNet")
        splitter = get_splitter("LPO")
        split = splitter(synthetic_dataset, n_splits=2, validation_ratio=0.4)[0]

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

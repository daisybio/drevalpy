"""Deterministic mocked tests for Ray/Optuna HPO."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
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


def _ray_state_fixture(monkeypatch) -> dict[str, int]:
    state = {"initialized": False, "init_calls": 0, "shutdown_calls": 0}

    def fake_init(**kwargs: Any) -> None:
        _ = kwargs
        state["initialized"] = True
        state["init_calls"] += 1

    def fake_shutdown() -> None:
        state["initialized"] = False
        state["shutdown_calls"] += 1

    monkeypatch.setattr("ray.init", fake_init)
    monkeypatch.setattr("ray.is_initialized", lambda: state["initialized"])
    monkeypatch.setattr("ray.shutdown", fake_shutdown)
    return state


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


def test_hpam_tune_zero_trials_skips_ray(monkeypatch) -> None:
    pytest.importorskip("ray")
    init_calls: list[dict[str, Any]] = []
    monkeypatch.setattr("ray.init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr("ray.is_initialized", lambda: False)

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
    assert init_calls == []


def test_hpam_tune_one_trial(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    captured: dict[str, int] = {}

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            _ = trainable, param_space, run_config
            captured["num_samples"] = tune_config.num_samples

        def fit(self):
            class Result:
                config = {"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = {"RMSE": 0.1}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

                @staticmethod
                def __iter__():
                    return iter([])

            return Results()

    state = _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)

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
    assert captured["num_samples"] == 1
    assert "alpha" in best
    assert state["init_calls"] == 1
    assert state["shutdown_calls"] == 1


def test_hpam_tune_all_nan_returns_defaults(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class FakeTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            class Result:
                config = {"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = {"RMSE": float("nan")}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

                def __iter__(self):
                    return iter([Result()])

            return Results()

    _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)

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
    assert best == model_cls.get_default_hyperparameters()


def test_hpam_tune_trial_exception_reports_nan(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class FakeTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            class FailedResult:
                config = {"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = None

            class GoodResult:
                config = {"predictor.elasticNet.alpha": 0.3, "predictor.elasticNet.l1_ratio": 0.3}
                metrics = {"RMSE": 0.05}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    raise RuntimeError("no best result")

                def __iter__(self):
                    return iter([FailedResult(), GoodResult()])

            return Results()

    _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)

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
    assert best["alpha"] == pytest.approx(0.3)


def test_hpam_tune_tuner_exception_cleans_up(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class BadTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            raise RuntimeError("tuner exploded")

    state = _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", BadTuner)

    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    with pytest.raises(RuntimeError, match="tuner exploded"):
        mu_hpam_tune(
            model_class=model_cls,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
        )
    assert state["shutdown_calls"] == 1


def test_hpam_tune_does_not_shutdown_preexisting_ray(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            pass

        def fit(self):
            class Result:
                config = {"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = {"RMSE": 0.1}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

                @staticmethod
                def __iter__():
                    return iter([])

            return Results()

    state: dict[str, Any] = {"initialized": True, "init_calls": 0, "shutdown_calls": 0}

    def fake_init(**kwargs):
        state["init_calls"] += 1

    def fake_shutdown():
        state["shutdown_calls"] += 1

    monkeypatch.setattr("ray.init", fake_init)
    monkeypatch.setattr("ray.is_initialized", lambda: True)
    monkeypatch.setattr("ray.shutdown", fake_shutdown)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)

    model_cls = construct_model("ElasticNet")
    mudataset, train_scope, val_scope = _tiny_mudataset_and_scopes()
    mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
    )
    assert state["init_calls"] == 0
    assert state["shutdown_calls"] == 0


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


@pytest.mark.skipif(os.environ.get("DREVALPY_RUN_RAY_TESTS") != "1", reason="optional Ray runtime test")
def test_hpam_tune_real_one_trial(tmp_path, data_dir) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    from drevalpy.data import load
    from drevalpy.data.splitters import get_splitter

    mudataset = load("TOYv1")
    splitter = get_splitter("LPO")
    folds = splitter.split(mudataset, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    from drevalpy.experiment.fold import prepare_mu_fold

    model_cls = construct_model("ElasticNet")
    fold_data = prepare_mu_fold(mudataset, split, model_cls)

    from drevalpy import experiment

    best = experiment.mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=fold_data.train_scope,
        val_scope=fold_data.val_scope,
        early_stopping_scope=fold_data.early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=1, storage_path=str(tmp_path)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

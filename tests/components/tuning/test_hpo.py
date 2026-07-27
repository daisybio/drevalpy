"""Deterministic mocked tests for Ray/Optuna HPO."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from drevalpy.components.tuning.config import HPOConfig, build_experiment_hpo_config
from drevalpy.components.tuning.hpo import hpam_tune
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import construct_model


def _tiny_dataset() -> DrugResponseDataset:
    return DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1"]),
    )


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
        "drevalpy.components.tuning.hpo.structured_space_for_drp_model",
        lambda _cls: {},
    )
    monkeypatch.setattr(
        "drevalpy.components.tuning.hpo.has_tunable_hyperparameters",
        lambda _cls: False,
    )

    dataset = _tiny_dataset()
    best = hpam_tune(
        model_class=model_cls,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
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
    dataset = _tiny_dataset()
    best = hpam_tune(
        model_class=model_cls,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
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
    monkeypatch.setattr(
        "drevalpy.experiment.train_and_evaluate",
        lambda **_kwargs: {"RMSE": 0.1},
    )

    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    best = hpam_tune(
        model_class=model_cls,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
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
                config = {"predictor.elasticNet.alpha": 0.5}
                metrics = {"RMSE": float("nan")}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

                @staticmethod
                def __iter__():
                    return iter([Result()])

            return Results()

    _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)
    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    with pytest.warns(UserWarning, match="did not find a valid configuration"):
        best = hpam_tune(
            model_class=model_cls,
            train_dataset=dataset,
            validation_dataset=dataset.copy(),
            early_stopping_dataset=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        )
    assert best == model_cls.get_default_hyperparameters()


def test_hpam_tune_trial_exception_reports_nan(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    reports: list[dict[str, float]] = []

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            _ = param_space, tune_config, run_config
            trainable({"predictor.elasticNet.alpha": 0.5, "predictor.elasticNet.l1_ratio": 0.5})

        def fit(self):
            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    raise RuntimeError("no successful trials")

                @staticmethod
                def __iter__():
                    return iter([])

            return Results()

    _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)
    monkeypatch.setattr("ray.tune.report", lambda metrics: reports.append(metrics))
    monkeypatch.setattr(
        "drevalpy.experiment.train_and_evaluate",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    with pytest.warns(UserWarning, match="did not find a valid configuration"):
        best = hpam_tune(
            model_class=model_cls,
            train_dataset=dataset,
            validation_dataset=dataset.copy(),
            early_stopping_dataset=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
        )
    assert len(reports) == 1
    assert np.isnan(reports[0]["RMSE"])
    assert best == model_cls.get_default_hyperparameters()


def test_hpam_tune_tuner_exception_cleans_up(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class BrokenTuner:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            raise RuntimeError("tuner failed")

    state = _ray_state_fixture(monkeypatch)
    monkeypatch.setattr("ray.tune.Tuner", BrokenTuner)
    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    with pytest.raises(RuntimeError, match="tuner failed"):
        hpam_tune(
            model_class=model_cls,
            train_dataset=dataset,
            validation_dataset=dataset.copy(),
            early_stopping_dataset=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        )
    assert state["shutdown_calls"] == 1


def test_hpam_tune_does_not_shutdown_preexisting_ray(monkeypatch) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    class FakeTuner:
        def __init__(self, *_args, **_kwargs):
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

    state = _ray_state_fixture(monkeypatch)
    state["initialized"] = True
    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)
    monkeypatch.setattr(
        "drevalpy.experiment.train_and_evaluate",
        lambda **_kwargs: {"RMSE": 0.1},
    )

    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    hpam_tune(
        model_class=model_cls,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
    )
    assert state["init_calls"] == 0
    assert state["shutdown_calls"] == 0


def test_hpam_tune_rejects_metric_mismatch() -> None:
    model_cls = construct_model("ElasticNet")
    dataset = _tiny_dataset()
    with pytest.raises(ValueError, match="must match metric argument"):
        hpam_tune(
            model_class=model_cls,
            train_dataset=dataset,
            validation_dataset=dataset.copy(),
            early_stopping_dataset=None,
            metric="RMSE",
            hpo_config=HPOConfig.from_metric("Pearson", n_trials=1),
        )


def test_build_experiment_hpo_config_matches_cv_and_final() -> None:
    cfg = build_experiment_hpo_config(
        "RMSE",
        n_trials=8,
        random_state=7,
        resources_per_trial={"cpu": 2},
        storage_path="raytune-storage",
    )
    assert cfg.n_trials == 8
    assert cfg.random_state == 7
    assert cfg.resources_per_trial == {"cpu": 2}
    assert cfg.storage_path == "raytune-storage"
    assert cfg.mode == "min"


def test_run_hpam_split_writes_single_default_yaml(tmp_path, monkeypatch) -> None:
    from drevalpy.cli.run_cv import run_hpam_split

    monkeypatch.chdir(tmp_path)
    run_hpam_split(model_name="ElasticNet", hyperparameter_tuning=False)
    assert list(tmp_path.glob("hpam_*.yaml")) == [tmp_path / "hpam_0.yaml"]


@pytest.mark.skipif(os.environ.get("DREVALPY_RUN_RAY_TESTS") != "1", reason="optional Ray runtime test")
def test_hpam_tune_real_one_trial(tmp_path, data_dir) -> None:
    pytest.importorskip("ray")
    pytest.importorskip("optuna")
    from drevalpy import experiment

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

    storage = tmp_path / "ray_storage"
    best = experiment.hpam_tune(
        model_class=model_cls,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        early_stopping_dataset=None,
        metric="RMSE",
        path_data=str(data_dir),
        hpo_config=build_experiment_hpo_config("RMSE", n_trials=1, storage_path=str(storage)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best

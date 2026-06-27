"""Wandb payload tests for Ray/Optuna HPO."""

from __future__ import annotations

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.models import MODEL_FACTORY


def test_hpam_tune_ray_optuna_logs_wandb_config(monkeypatch) -> None:
    import pytest

    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    init_calls: list[dict] = []

    class FakeWandb:
        run = object()

        @staticmethod
        def init(**kwargs):
            init_calls.append(kwargs)

        @staticmethod
        def finish():
            return None

        @staticmethod
        def config_update(*_args, **_kwargs):
            return None

    monkeypatch.setitem(__import__("sys").modules, "wandb", FakeWandb)

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            trainable(
                {
                    "predictor.elasticNet.alpha": 0.2,
                    "predictor.elasticNet.l1_ratio": 0.5,
                }
            )

        def fit(self):
            class Result:
                config = {"predictor.elasticNet.alpha": 0.2, "predictor.elasticNet.l1_ratio": 0.5}
                metrics = {"RMSE": 0.2}

            class Results:
                @staticmethod
                def get_best_result(*_args, **_kwargs):
                    return Result()

            return Results()

    monkeypatch.setattr("ray.tune.Tuner", FakeTuner)
    monkeypatch.setattr("ray.init", lambda **kwargs: None)

    from drevalpy.components.tuning.hpo import hpam_tune_ray_optuna
    from drevalpy.datasets.dataset import DrugResponseDataset

    model_cls = MODEL_FACTORY["ElasticNet"]
    model = model_cls()
    dataset = DrugResponseDataset(
        response=[1.0, 2.0],
        cell_line_ids=["cl1", "cl2"],
        drug_ids=["d1", "d1"],
    )
    hpam_tune_ray_optuna(
        model=model,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
        model_class=model_cls,
        metric="RMSE",
        path_data="data",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=1),
        wandb_project="test-project",
        wandb_base_config={"dataset": "TOYv1"},
    )
    assert init_calls
    config = init_calls[0]["config"]
    assert config["phase"] == "hyperparameter_tuning"
    assert config["hpo_backend"] == "ray"
    assert config["search_alg"] == "optuna"
    assert "hyperparameters" in config

"""Wandb payload tests for Ray/Optuna HPO."""

from __future__ import annotations

import numpy as np

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.models import construct_model


def test_hpam_tune_ray_optuna_logs_wandb_config(monkeypatch) -> None:
    import pytest

    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    init_calls: list[dict] = []

    def fake_init_wandb(
        self,
        project: str,
        config: dict | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        finish_previous: bool = True,
    ) -> None:
        _ = name, tags, finish_previous
        init_calls.append({"project": project, "config": config})
        self.wandb_project = project
        self.wandb_run = object()

    def fake_is_wandb_enabled(self) -> bool:
        return self.wandb_project is not None

    def fake_finish_wandb(self) -> None:
        self.wandb_project = None
        self.wandb_run = None

    monkeypatch.setattr("drevalpy.models.drp_model.DRPModel.init_wandb", fake_init_wandb)
    monkeypatch.setattr("drevalpy.models.drp_model.DRPModel.is_wandb_enabled", fake_is_wandb_enabled)
    monkeypatch.setattr("drevalpy.models.drp_model.DRPModel.finish_wandb", fake_finish_wandb)
    monkeypatch.setattr(
        "drevalpy.experiment.train_and_evaluate",
        lambda **_kwargs: {"RMSE": 0.2},
    )

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
    monkeypatch.setattr("ray.is_initialized", lambda: True)

    from drevalpy.components.tuning.hpo import hpam_tune_ray_optuna
    from drevalpy.datasets.dataset import DrugResponseDataset

    model_cls = construct_model("ElasticNet")
    model = model_cls()
    dataset = DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1"]),
    )
    hpam_tune_ray_optuna(
        model=model,
        train_dataset=dataset,
        validation_dataset=dataset.copy(),
        early_stopping_dataset=None,
        model_class=model_cls,
        metric="RMSE",
        path_data="data",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        wandb_project="test-project",
        wandb_base_config={"dataset": "TOYv1"},
    )
    assert init_calls
    config = init_calls[0]["config"]
    assert config["phase"] == "hyperparameter_tuning"
    assert config["hpo_backend"] == "ray"
    assert config["search_alg"] == "optuna"
    assert "hyperparameters" in config

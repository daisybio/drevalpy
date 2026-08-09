"""Wandb payload tests for Ray/Optuna HPO."""

from __future__ import annotations

import numpy as np

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.data.splitting import EntityScope
from drevalpy.models import construct_model


def test_hpam_tune_logs_wandb_config(monkeypatch) -> None:
    import pytest

    pytest.importorskip("ray")
    pytest.importorskip("optuna")

    init_calls: list[dict] = []

    def fake_init_wandb(
        self,
        *,
        project: str | None = None,
        config: dict | None = None,
        reinit: bool = False,
        name: str | None = None,
    ):
        init_calls.append({"project": project, "config": config, "reinit": reinit, "name": name})

    class FakeTuner:
        def __init__(self, trainable, param_space, tune_config, run_config=None):
            pass

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

    from drevalpy.components.tuning.hpo import mu_hpam_tune
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    model_cls = construct_model("ElasticNet")
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    train_scope = EntityScope(cell_lines=np.array([0, 1]), drugs=np.array([0, 1]))
    val_scope = EntityScope(cell_lines=np.array([0, 1]), drugs=np.array([0, 1]))
    mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        wandb_project="test-project",
        wandb_base_config={"dataset": "TOYv1"},
    )

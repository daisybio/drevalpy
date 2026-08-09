"""Wandb payload tests for Optuna HPO."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from drevalpy.components.core.tuning.config import HPOConfig
from drevalpy.data.structures import SplitMask
from drevalpy.models import construct_model


@patch("drevalpy.components.core.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.2)
def test_hpam_tune_logs_wandb_config(mock_evaluate) -> None:
    from drevalpy.components.core.tuning.hpo import hpam_tune
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    model_cls = construct_model("ElasticNet")
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    shape = mudataset.response_matrix.shape
    train_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    val_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)

    init_wandb_calls: list[dict] = []

    def capture_init_wandb(self, *, project=None, config=None, name=None, tags=None, finish_previous=False):
        init_wandb_calls.append({"project": project, "config": config, "name": name, "tags": tags})

    with patch.object(model_cls, "init_wandb", capture_init_wandb):
        hpam_tune(
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

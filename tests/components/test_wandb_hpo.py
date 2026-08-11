"""Wandb payload tests for Optuna HPO."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from drevalpy.models import construct_model
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.types import SplitMask


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.2)
@patch("drevalpy.models.tuning.hpo._log_trial_to_wandb")
def test_hpam_tune_logs_wandb_config(mock_wandb_log, mock_evaluate) -> None:
    from drevalpy.models.tuning.hpo import hpam_tune
    from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

    model_cls = construct_model("ElasticNet")
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    shape = mudataset.response_matrix.shape
    train_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)
    val_scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=shape)

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

    assert mock_wandb_log.call_count > 0
    call_kwargs = mock_wandb_log.call_args_list[0].kwargs
    assert call_kwargs["wandb_project"] == "test-project"

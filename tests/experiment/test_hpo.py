"""Tests for experiment_hpo hyperparameter selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from drevalpy.data.structures.splitting import EntityScope
from drevalpy.experiment.hpo import select_fold_hyperparameters
from drevalpy.models._model_lookup import get_model_class


def _dummy_scope() -> EntityScope:
    return EntityScope(
        cell_lines=np.arange(5),
        drugs=None,
    )


@patch("drevalpy.components.tuning.hpo.mu_hpam_tune")
@patch(
    "drevalpy.components.tuning.drp_hyperparameters.has_tunable_hyperparameters",
    return_value=True,
)
def test_select_fold_hyperparameters_tunes_when_enabled(_has_tunable, mock_tune) -> None:
    """Verify that HPO is invoked when tuning is enabled."""
    mock_tune.return_value = {"alpha": 0.5}
    model_class = get_model_class("ElasticNet")
    mudataset = MagicMock()
    scope = _dummy_scope()
    result = select_fold_hyperparameters(
        model_class=model_class,
        mudataset=mudataset,
        train_scope=scope,
        val_scope=scope,
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir=None,
        hyperparameter_tuning=True,
        hpo_config=MagicMock(),
    )
    assert result == {"alpha": 0.5}
    mock_tune.assert_called_once()


@patch(
    "drevalpy.components.tuning.drp_hyperparameters.has_tunable_hyperparameters",
    return_value=True,
)
def test_select_fold_hyperparameters_defaults_when_tuning_off(_has_tunable) -> None:
    """Verify that defaults are returned when tuning is disabled."""
    model_class = get_model_class("ElasticNet")
    mudataset = MagicMock()
    scope = _dummy_scope()
    result = select_fold_hyperparameters(
        model_class=model_class,
        mudataset=mudataset,
        train_scope=scope,
        val_scope=scope,
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir=None,
        hyperparameter_tuning=False,
        hpo_config=MagicMock(),
    )
    assert result == model_class.get_default_hyperparameters()

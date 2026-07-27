"""Tests for experiment_hpo hyperparameter selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from drevalpy.experiment.hpo import select_fold_hyperparameters
from drevalpy.models._model_lookup import get_model_class


@patch("drevalpy.experiment.hpo.hpam_tune")
@patch(
    "drevalpy.components.tuning.drp_hyperparameters.has_tunable_hyperparameters",
    return_value=True,
)
def test_select_fold_hyperparameters_tunes_when_enabled(_has_tunable, mock_tune) -> None:
    mock_tune.return_value = {"alpha": 0.5}
    model_class = get_model_class("ElasticNet")
    train = MagicMock()
    val = MagicMock()
    result = select_fold_hyperparameters(
        model_class=model_class,
        train_dataset=train,
        validation_dataset=val,
        early_stopping_dataset=None,
        response_transformation=None,
        metric="RMSE",
        path_data="data",
        model_checkpoint_dir="TEMPORARY",
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
    model_class = get_model_class("ElasticNet")
    result = select_fold_hyperparameters(
        model_class=model_class,
        train_dataset=MagicMock(),
        validation_dataset=MagicMock(),
        early_stopping_dataset=None,
        response_transformation=None,
        metric="RMSE",
        path_data="data",
        model_checkpoint_dir="TEMPORARY",
        hyperparameter_tuning=False,
        hpo_config=MagicMock(),
    )
    assert result == model_class.get_default_hyperparameters()

"""Tests for Optuna HPO runtime helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import optuna

from drevalpy.models import construct_model
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo_runtime import (
    _construct_trial_model,
    _init_trial_wandb,
    _optuna_objective,
    _wandb_trial_run_config,
    _wandb_trial_run_name,
    run_optuna_study,
)


@patch("drevalpy.models.tuning.hpo_runtime.tuned_config_for_drp_model", return_value=None)
def test_construct_trial_model_without_tuned_config(mock_tuned_config) -> None:
    model_class = construct_model("ElasticNet")
    sampled = {"alpha": 0.5}
    trial_model = _construct_trial_model(model_class, sampled)
    mock_tuned_config.assert_called_once_with(model_class, sampled)
    assert trial_model.hyperparameters["alpha"] == 0.5


@patch("drevalpy.models.tuning.hpo_runtime.construct_drp_model_from_config")
@patch("drevalpy.models.tuning.hpo_runtime.tuned_config_for_drp_model")
def test_construct_trial_model_with_tuned_config(mock_tuned_config, mock_construct) -> None:
    model_class = construct_model("ElasticNet")
    sampled = {"alpha": 0.5}
    tuned_config = MagicMock()
    expected_model = MagicMock()
    mock_tuned_config.return_value = tuned_config
    mock_construct.return_value = expected_model

    trial_model = _construct_trial_model(model_class, sampled)

    mock_construct.assert_called_once_with(model_class, tuned_config)
    assert trial_model is expected_model


def test_wandb_trial_run_config_merges_base_config() -> None:
    trial_model = MagicMock()
    trial_model.hyperparameters = {"alpha": 0.1}
    cfg = HPOConfig.from_metric("RMSE", n_trials=3)

    config = _wandb_trial_run_config(
        trial_model=trial_model,
        cfg=cfg,
        wandb_base_config={"dataset": "GDSC1"},
        trial_number=1,
    )

    assert config["dataset"] == "GDSC1"
    assert config["phase"] == "hyperparameter_tuning"
    assert config["hpo_backend"] == "optuna"
    assert config["trial_number"] == 1
    assert config["hyperparameters"] == {"alpha": 0.1}


def test_wandb_trial_run_name_includes_split_and_trial() -> None:
    assert _wandb_trial_run_name(model_name="ElasticNet", split_index=2, trial_number=5) == (
        "ElasticNet_split_2_trial_5"
    )
    assert _wandb_trial_run_name(model_name="ElasticNet", split_index=None, trial_number=3) == ("ElasticNet_trial_3")


@patch("drevalpy.models.tuning.hpo_runtime._wandb_trial_run_name", return_value="run-name")
@patch("drevalpy.models.tuning.hpo_runtime._wandb_trial_run_config", return_value={"trial_number": 7})
def test_init_trial_wandb_delegates_to_model(mock_run_config, mock_run_name) -> None:
    trial_model = MagicMock()
    cfg = HPOConfig.from_metric("RMSE")

    _init_trial_wandb(
        trial_model,
        wandb_project="dreval-hpo",
        wandb_base_config={"fold": 1},
        cfg=cfg,
        model_name="ElasticNet",
        split_index=0,
        trial_number=7,
    )

    mock_run_config.assert_called_once()
    mock_run_name.assert_called_once_with(model_name="ElasticNet", split_index=0, trial_number=7)
    trial_model.init_wandb.assert_called_once_with(
        project="dreval-hpo",
        config={"trial_number": 7},
        name="run-name",
        tags=["ElasticNet", "hpam_tuning", "optuna"],
        finish_previous=True,
    )


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.33)
@patch("drevalpy.models.tuning.hpo_runtime._construct_trial_model")
@patch("drevalpy.models.tuning.hpo_runtime.sample_from_optuna_trial")
def test_optuna_objective_returns_score(
    mock_sample,
    mock_construct,
    mock_evaluate,
) -> None:
    trial_model = MagicMock()
    mock_construct.return_value = trial_model
    mock_sample.return_value = {"alpha": 0.1}

    study = optuna.create_study()
    trial = study.ask()

    score = _optuna_objective(
        trial,
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        structured_space={"alpha": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.5}},
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project=None,
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    assert score == 0.33
    mock_construct.assert_called_once()
    mock_evaluate.assert_called_once()


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", side_effect=RuntimeError("boom"))
@patch("drevalpy.models.tuning.hpo_runtime._construct_trial_model")
@patch("drevalpy.models.tuning.hpo_runtime.sample_from_optuna_trial")
def test_optuna_objective_failure_returns_nan(mock_sample, mock_construct, mock_evaluate) -> None:
    mock_construct.return_value = MagicMock()
    mock_sample.return_value = {"alpha": 0.1}

    study = optuna.create_study()
    trial = study.ask()

    score = _optuna_objective(
        trial,
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        structured_space={"alpha": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.5}},
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project=None,
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    assert score != score  # NaN


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.5)
@patch("drevalpy.models.tuning.hpo_runtime._init_trial_wandb")
@patch("drevalpy.models.tuning.hpo_runtime._construct_trial_model")
@patch("drevalpy.models.tuning.hpo_runtime.sample_from_optuna_trial")
def test_optuna_objective_wandb_finishes_in_finally(
    mock_sample,
    mock_construct,
    mock_init_wandb,
    mock_evaluate,
) -> None:
    trial_model = MagicMock()
    trial_model.is_wandb_enabled.return_value = True
    mock_construct.return_value = trial_model
    mock_sample.return_value = {"alpha": 0.1}

    study = optuna.create_study()
    trial = study.ask()

    score = _optuna_objective(
        trial,
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        structured_space={"alpha": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.5}},
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project="dreval-hpo",
        wandb_base_config=None,
        split_index=1,
        model_name="ElasticNet",
    )

    assert score == 0.5
    mock_init_wandb.assert_called_once()
    trial_model.finish_wandb.assert_called_once()


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", side_effect=RuntimeError("boom"))
@patch("drevalpy.models.tuning.hpo_runtime._init_trial_wandb")
@patch("drevalpy.models.tuning.hpo_runtime._construct_trial_model")
@patch("drevalpy.models.tuning.hpo_runtime.sample_from_optuna_trial")
def test_optuna_objective_wandb_failure_still_finishes(
    mock_sample,
    mock_construct,
    mock_init_wandb,
    mock_evaluate,
) -> None:
    trial_model = MagicMock()
    trial_model.is_wandb_enabled.return_value = True
    mock_construct.return_value = trial_model
    mock_sample.return_value = {"alpha": 0.1}

    study = optuna.create_study()
    trial = study.ask()

    score = _optuna_objective(
        trial,
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        structured_space={"alpha": {"type": "float", "low": 0.01, "high": 1.0, "default": 0.5}},
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project="dreval-hpo",
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    assert score != score  # NaN
    trial_model.finish_wandb.assert_called_once()


def test_run_optuna_study_uses_tpe_sampler() -> None:
    cfg = HPOConfig.from_metric("RMSE", n_trials=3)

    def dummy_objective(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        return 0.5

    study = run_optuna_study(objective=dummy_objective, cfg=cfg)
    assert len(study.trials) == 3
    assert isinstance(study.sampler, optuna.samplers.TPESampler)

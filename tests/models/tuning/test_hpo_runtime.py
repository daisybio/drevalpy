"""Tests for HPO trial construction with immutable ModelSpec inputs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import optuna
import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo_runtime import (
    _construct_trial_model,
    _optuna_objective,
    _wandb_trial_run_config,
    _wandb_trial_run_name,
)


def test_construct_trial_model_keeps_spec_identity() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    trial_model = _construct_trial_model(spec, {"predictor.elasticNet.alpha": 0.5})

    assert trial_model.spec.name == spec.name
    assert trial_model.hyperparameters["alpha"] == 0.5


def test_wandb_name_includes_split_and_trial() -> None:
    assert _wandb_trial_run_name(model_name="ElasticNet", split_index=2, trial_number=5) == (
        "ElasticNet_split_2_trial_5"
    )


def test_wandb_config_contains_model_hyperparameters() -> None:
    model = MagicMock()
    model.hyperparameters = {"alpha": 0.1}
    config = _wandb_trial_run_config(
        trial_model=model,
        cfg=HPOConfig.from_metric("RMSE", n_trials=3),
        wandb_base_config={"dataset": "synthetic"},
        trial_number=1,
    )

    assert config["dataset"] == "synthetic"
    assert config["hyperparameters"] == {"alpha": 0.1}


@patch("drevalpy.models.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.33)
@patch("drevalpy.models.tuning.hpo_runtime._construct_trial_model")
@patch("drevalpy.models.tuning.hpo_runtime.sample_from_optuna_trial", return_value={"alpha": 0.1})
def test_objective_accepts_model_spec(mock_sample, mock_construct, mock_evaluate) -> None:
    spec = ModelSpec.from_name("ElasticNet")
    mock_construct.return_value = MagicMock()
    study = optuna.create_study()
    trial = study.ask()

    score = _optuna_objective(
        trial,
        model_spec=spec,
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
        model_name=spec.name,
    )

    assert score == 0.33
    mock_construct.assert_called_once_with(spec, {"alpha": 0.1}, None)
    mock_evaluate.assert_called_once()


@pytest.mark.parametrize("supplied", [False, True])
def test_trial_uses_one_training_call_and_keeps_nan_scope_alignment(tmp_path, supplied) -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lco_split_masks, synthetic_mudataset_identity

    dataset = synthetic_mudataset_identity()
    scopes = lco_split_masks()
    model = MagicMock()
    model.predict.return_value = np.array([np.nan, 5.0])
    metrics, predictions = _mu_evaluate_trial_all_metrics(
        model,
        mudataset=dataset,
        train_scope=scopes.train,
        val_scope=scopes.test,
        early_stopping_scope=None,
        model_checkpoint_dir=tmp_path if supplied else None,
        trial_number=3,
    )
    model.train.assert_called_once()
    assert model.train.call_args.args == (dataset, scopes.train)
    assert model.train.call_args.kwargs["early_stopping_scope"] is None
    if supplied:
        assert model.train.call_args.kwargs["model_checkpoint_dir"] == tmp_path / "trial_3"
    assert metrics["MSE"] == 1.0
    assert np.isnan(metrics["R^2"])
    np.testing.assert_array_equal(predictions, [np.nan, 5.0])


def test_trial_rejects_prediction_truncation() -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lco_split_masks, synthetic_mudataset_identity

    model = MagicMock()
    model.predict.return_value = np.array([1.0])
    scopes = lco_split_masks()
    with pytest.raises(ValueError, match="shape"):
        _mu_evaluate_trial_all_metrics(
            model,
            mudataset=synthetic_mudataset_identity(),
            train_scope=scopes.train,
            val_scope=scopes.test,
            early_stopping_scope=None,
            model_checkpoint_dir=None,
        )


def test_trial_with_no_joint_valid_pairs_keeps_empty_metric_dictionary() -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lco_split_masks, synthetic_mudataset_identity

    model = MagicMock()
    model.predict.return_value = np.full(2, np.nan)
    scopes = lco_split_masks()
    metrics, predictions = _mu_evaluate_trial_all_metrics(
        model,
        mudataset=synthetic_mudataset_identity(),
        train_scope=scopes.train,
        val_scope=scopes.test,
        early_stopping_scope=None,
        model_checkpoint_dir=None,
    )
    assert metrics == {}
    assert predictions.shape == (2,)

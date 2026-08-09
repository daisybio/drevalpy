"""Tests for Ray HPO runtime helpers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.hpo import _select_best_result
from drevalpy.components.tuning.hpo_runtime import (
    _construct_trial_model,
    _init_trial_wandb,
    _report_trial_failure,
    _report_trial_score,
    _wandb_trial_run_config,
    _wandb_trial_run_name,
    mu_build_ray_trainable,
)
from drevalpy.models import construct_model


class _FakeResult:
    def __init__(self, metrics: dict[str, float], config: dict[str, object] | None) -> None:
        self.metrics = metrics
        self.config = config


class _FakeResults:
    def __init__(self, trials: list[_FakeResult], *, raise_on_best: bool = False) -> None:
        self._trials = trials
        self._raise_on_best = raise_on_best

    def get_best_result(self, *, metric: str, mode: str) -> _FakeResult:
        _ = metric, mode
        if self._raise_on_best:
            raise RuntimeError("ray best failed")
        return self._trials[0]

    def __iter__(self):
        return iter(self._trials)


def test_select_best_result_prefers_ray_when_usable() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults(
        [
            _FakeResult({"RMSE": 1.0}, {"alpha": 0.1}),
            _FakeResult({"RMSE": 0.5}, {"alpha": 0.2}),
        ]
    )
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"alpha": 0.1}


def test_select_best_result_scans_when_ray_best_is_nan() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults(
        [
            _FakeResult({"RMSE": float("nan")}, {"alpha": 0.1}),
            _FakeResult({"RMSE": 0.4}, {"alpha": 0.2}),
        ]
    )
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"alpha": 0.2}


def test_select_best_result_max_mode_picks_higher_score() -> None:
    cfg = HPOConfig.from_metric("Pearson", n_trials=2)
    results = _FakeResults([], raise_on_best=True)
    results._trials = [
        _FakeResult({"Pearson": 0.2}, {"a": 1}),
        _FakeResult({"Pearson": 0.9}, {"a": 2}),
    ]
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"a": 2}


def test_select_best_result_returns_none_when_all_invalid() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults([], raise_on_best=True)
    results._trials = [
        _FakeResult({"RMSE": float("nan")}, {"a": 1}),
        _FakeResult({"RMSE": 1.0}, None),
    ]
    assert _select_best_result(results, cfg) is None


@patch("drevalpy.components.tuning.hpo_runtime.tuned_config_for_drp_model", return_value=None)
def test_construct_trial_model_without_tuned_config(mock_tuned_config) -> None:
    model_class = construct_model("ElasticNet")
    sampled = {"alpha": 0.5}
    trial_model = _construct_trial_model(model_class, sampled)
    mock_tuned_config.assert_called_once_with(model_class, sampled)
    assert trial_model.hyperparameters["alpha"] == 0.5


@patch("drevalpy.components.tuning.hpo_runtime.construct_drp_model_from_config")
@patch("drevalpy.components.tuning.hpo_runtime.tuned_config_for_drp_model")
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


def test_report_trial_score_reports_metric() -> None:
    """Stub ray in sys.modules so the test works when Ray has no Windows/3.13 wheel."""
    fake_tune = MagicMock()
    fake_ray = MagicMock()
    fake_ray.tune = fake_tune
    with patch.dict(sys.modules, {"ray": fake_ray, "ray.tune": fake_tune}):
        _report_trial_score("RMSE", 0.25)
    fake_tune.report.assert_called_once_with({"RMSE": 0.25})


@patch("drevalpy.components.tuning.hpo_runtime._report_trial_score")
def test_report_trial_failure_reports_nan(mock_report_score) -> None:
    _report_trial_failure("RMSE")
    mock_report_score.assert_called_once()
    metric, score = mock_report_score.call_args.args
    assert metric == "RMSE"
    assert score != score  # NaN


def test_wandb_trial_run_config_merges_base_config() -> None:
    trial_model = MagicMock()
    trial_model.hyperparameters = {"alpha": 0.1}
    cfg = HPOConfig.from_metric("RMSE", n_trials=3)

    config = _wandb_trial_run_config(
        trial_model=trial_model,
        cfg=cfg,
        wandb_base_config={"dataset": "GDSC1"},
        trial_id="trial-1",
    )

    assert config["dataset"] == "GDSC1"
    assert config["phase"] == "hyperparameter_tuning"
    assert config["hpo_backend"] == "ray"
    assert config["trial_id"] == "trial-1"
    assert config["hyperparameters"] == {"alpha": 0.1}


def test_wandb_trial_run_name_includes_split_and_trial() -> None:
    assert _wandb_trial_run_name(model_name="ElasticNet", split_index=2, trial_id="abc") == (
        "ElasticNet_split_2_trial_abc"
    )
    assert _wandb_trial_run_name(model_name="ElasticNet", split_index=None, trial_id="abc") == ("ElasticNet_trial_abc")


@patch("drevalpy.components.tuning.hpo_runtime.current_trial_id", return_value="trial-7")
@patch("drevalpy.components.tuning.hpo_runtime._wandb_trial_run_name", return_value="run-name")
@patch("drevalpy.components.tuning.hpo_runtime._wandb_trial_run_config", return_value={"trial_id": "trial-7"})
def test_init_trial_wandb_delegates_to_model(mock_run_config, mock_run_name, mock_trial_id) -> None:
    trial_model = MagicMock()
    cfg = HPOConfig.from_metric("RMSE")

    _init_trial_wandb(
        trial_model,
        wandb_project="dreval-hpo",
        wandb_base_config={"fold": 1},
        cfg=cfg,
        model_name="ElasticNet",
        split_index=0,
    )

    mock_trial_id.assert_called_once()
    mock_run_config.assert_called_once()
    mock_run_name.assert_called_once_with(model_name="ElasticNet", split_index=0, trial_id="trial-7")
    trial_model.init_wandb.assert_called_once_with(
        project="dreval-hpo",
        config={"trial_id": "trial-7"},
        name="run-name",
        tags=["ElasticNet", "hpam_tuning", "ray", "optuna"],
        finish_previous=True,
    )


@patch("drevalpy.components.tuning.hpo_runtime._report_trial_score")
@patch("drevalpy.components.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.33)
@patch("drevalpy.components.tuning.hpo_runtime._construct_trial_model")
def test_mu_build_ray_trainable_success_reports_score(
    mock_construct,
    mock_evaluate,
    mock_report_score,
) -> None:
    trial_model = MagicMock()
    mock_construct.return_value = trial_model
    trainable = mu_build_ray_trainable(
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project=None,
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    trainable({"alpha": 0.1})

    mock_construct.assert_called_once()
    mock_evaluate.assert_called_once()
    mock_report_score.assert_called_once_with("RMSE", 0.33)


@patch("drevalpy.components.tuning.hpo_runtime._report_trial_failure")
@patch("drevalpy.components.tuning.hpo_runtime._mu_evaluate_trial_model", side_effect=RuntimeError("boom"))
@patch("drevalpy.components.tuning.hpo_runtime._construct_trial_model")
def test_mu_build_ray_trainable_failure_reports_nan(mock_construct, mock_evaluate, mock_report_failure) -> None:
    mock_construct.return_value = MagicMock()
    trainable = mu_build_ray_trainable(
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project=None,
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    trainable({"alpha": 0.1})

    mock_report_failure.assert_called_once_with("RMSE")


@patch("drevalpy.components.tuning.hpo_runtime._report_trial_score")
@patch("drevalpy.components.tuning.hpo_runtime._mu_evaluate_trial_model", return_value=0.5)
@patch("drevalpy.components.tuning.hpo_runtime._init_trial_wandb")
@patch("drevalpy.components.tuning.hpo_runtime._construct_trial_model")
def test_mu_build_ray_trainable_wandb_finishes_in_finally(
    mock_construct,
    mock_init_wandb,
    mock_evaluate,
    mock_report_score,
) -> None:
    trial_model = MagicMock()
    trial_model.is_wandb_enabled.return_value = True
    mock_construct.return_value = trial_model
    trainable = mu_build_ray_trainable(
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project="dreval-hpo",
        wandb_base_config=None,
        split_index=1,
        model_name="ElasticNet",
    )

    trainable({"alpha": 0.1})

    mock_init_wandb.assert_called_once()
    mock_report_score.assert_called_once_with("RMSE", 0.5)
    trial_model.finish_wandb.assert_called_once()


@patch("drevalpy.components.tuning.hpo_runtime._report_trial_failure")
@patch("drevalpy.components.tuning.hpo_runtime._mu_evaluate_trial_model", side_effect=RuntimeError("boom"))
@patch("drevalpy.components.tuning.hpo_runtime._init_trial_wandb")
@patch("drevalpy.components.tuning.hpo_runtime._construct_trial_model")
def test_mu_build_ray_trainable_wandb_failure_still_finishes(
    mock_construct,
    mock_init_wandb,
    mock_evaluate,
    mock_report_failure,
) -> None:
    trial_model = MagicMock()
    trial_model.is_wandb_enabled.return_value = True
    mock_construct.return_value = trial_model
    trainable = mu_build_ray_trainable(
        model_class=construct_model("ElasticNet"),
        mudataset=MagicMock(),
        train_scope=MagicMock(),
        val_scope=MagicMock(),
        early_stopping_scope=None,
        response_transformation=None,
        metric="RMSE",
        model_checkpoint_dir="checkpoints",
        cfg=HPOConfig.from_metric("RMSE"),
        wandb_project="dreval-hpo",
        wandb_base_config=None,
        split_index=None,
        model_name="ElasticNet",
    )

    trainable({"alpha": 0.1})

    mock_report_failure.assert_called_once_with("RMSE")
    trial_model.finish_wandb.assert_called_once()

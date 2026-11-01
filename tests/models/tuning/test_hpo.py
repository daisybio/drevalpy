"""HPO: the trial loop reading its settings from the execution, and W&B tracking of trials."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.errors import DrEvalError
from drevalpy.evaluation import Metric
from drevalpy.experiment._execution import ExecutionSpec, FitScopes
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import from_recipe
from drevalpy.models.tuning import hpo
from drevalpy.models.tuning.hpo import HPOTrialsFailedError, hpam_tune
from drevalpy.types import PairMask
from drevalpy.types.results._identity import RunIdentity
from tests._fake_wandb import FakeWandb, install_fake_wandb
from tests._import_shims import block_imports
from tests.models.synthetic_fixtures import synthetic_model_dataset_gene_expression_fingerprints

SRC = Path(__file__).resolve().parents[3] / "src" / "drevalpy"


def _execution(
    spec: ModelSpec, n_trials: int, *, tuning: bool = True, metric: str = "RMSE", fold_index: int = 0
) -> ExecutionSpec:
    scope = PairMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=(2, 2))
    return ExecutionSpec(
        model=spec,
        scopes=FitScopes(scope, None, scope, scope, scope),
        identity=RunIdentity(spec.name, "synthetic", "LPO", fold_index),
        hyperparameter_tuning=tuning,
        hpo_metric=metric,
        hpo_num_trials=n_trials,
        hpo_random_state=42,
        seed=42,
        metadata={},
    )


def _tune(spec: ModelSpec, n_trials: int, *, tuning: bool = True, metric: str = "RMSE", fold_index: int = 0, **kwargs):
    execution = _execution(spec, n_trials, tuning=tuning, metric=metric, fold_index=fold_index)
    return hpam_tune(execution, synthetic_model_dataset_gene_expression_fingerprints(), **kwargs)


@pytest.mark.parametrize(
    ("name", "n_trials", "tuning"),
    [("ElasticNet", 0, True), ("ElasticNet", 2, False), ("NaivePredictor", 2, True)],
    ids=["zero-trials", "untuned", "empty-space"],
)
def test_nothing_to_tune_returns_the_spec_values(name: str, n_trials: int, tuning: bool) -> None:
    spec = ModelSpec.from_name(name).with_hyperparameters({"alpha": 0.3} if name == "ElasticNet" else {})

    with patch.object(DRPModel, "train", side_effect=AssertionError("no trial may train")):
        best, trials = _tune(spec, n_trials, tuning=tuning)

    assert best == dict(spec.hyperparameters)
    assert trials == []


def _scores(*rmse: float):
    """Patch ``evaluate`` so the trials score *rmse* in order, without fitting a model."""
    return patch.object(hpo, "evaluate", side_effect=[{"RMSE": value} for value in rmse])


@patch.object(DRPModel, "predict", return_value=np.zeros(4))
@patch.object(DRPModel, "train")
def test_each_trial_trains_on_the_training_scope_with_n_jobs(train, predict) -> None:
    with _scores(0.2, 0.1):
        best, trials = _tune(ModelSpec.from_name("ElasticNet"), 2, n_jobs=4)

    assert set(best) == set(ModelSpec.from_name("ElasticNet").hyperparameters)
    assert train.call_count == predict.call_count == 2
    assert [call.kwargs["n_jobs"] for call in train.call_args_list] == [4, 4]
    assert all(call.kwargs["early_stopping_scope"] is None for call in train.call_args_list)
    assert "model_checkpoint_dir" not in train.call_args.kwargs
    assert [trial.metrics for trial in trials] == [{"RMSE": 0.2}, {"RMSE": 0.1}]
    assert {trial.optimization_metric for trial in trials} == {"RMSE"}


@pytest.mark.parametrize("metric", list(Metric), ids=lambda metric: metric.replace("^", ""))
def test_the_study_direction_follows_the_execution_metric(metric: Metric, monkeypatch: pytest.MonkeyPatch) -> None:
    import optuna

    created = []
    real = optuna.create_study
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: created.append(kwargs) or real(**kwargs))

    with patch.object(hpo, "evaluate", return_value={metric.value: 0.5}):
        with patch.object(DRPModel, "train"), patch.object(DRPModel, "predict", return_value=np.zeros(4)):
            _tune(ModelSpec.from_name("ElasticNet"), 1, metric=metric.value)

    assert created[0]["direction"] == ("minimize" if metric.direction == "min" else "maximize")


@patch.object(DRPModel, "train", side_effect=RuntimeError("boom"))
def test_all_failed_trials_raise_without_default_fallback(train) -> None:
    with pytest.raises(HPOTrialsFailedError, match="All 2 hyperparameter trials failed"):
        _tune(ModelSpec.from_name("ElasticNet"), 2)
    assert issubclass(HPOTrialsFailedError, DrEvalError)
    assert issubclass(HPOTrialsFailedError, RuntimeError)


@patch.object(DRPModel, "predict", return_value=np.zeros(4))
@patch.object(DRPModel, "train")
def test_best_hyperparameters_carry_the_decoded_pow2_value_the_trials_trained_with(train, predict) -> None:
    spec = ModelSpec("FingerprintElasticNet", from_recipe("pca[gene_expression]:fingerprints:elasticNet"))

    with _scores(1.0, 0.9, 0.8, 0.7):
        best, trials = _tune(spec, 4)

    best_trial_params = trials[-1].hyperparameters
    assert best == {**spec.hyperparameters, **best_trial_params}
    assert best["drug_featurizer.fingerprints.n_bits"] in {512, 1024, 2048}
    assert all(key in spec.hyperparameter_space for key in best_trial_params)


def test_a_trial_keeps_nan_predictions_aligned_with_the_validation_scope() -> None:
    with patch.object(DRPModel, "train"), patch.object(DRPModel, "predict", return_value=np.array([np.nan, 5, 5, 5])):
        _, trials = _tune(ModelSpec.from_name("ElasticNet"), 1)

    np.testing.assert_array_equal(trials[0].predictions, [np.nan, 5, 5, 5])
    assert set(trials[0].metrics) >= {"RMSE", "MSE"}


def test_removed_legacy_hpo_artifacts_stay_removed() -> None:
    from drevalpy.models import drp_model

    assert "ParameterGrid" not in inspect.getsource(hpam_tune)
    assert "yaml" not in inspect.getsource(drp_model.DRPModel.__init__).lower()
    literature = SRC / "components" / "predictors" / "literature"
    assert not (literature / "impl").exists()
    assert not list(literature.glob("*_predictor.py"))
    assert not list(SRC.rglob("hyperparameters.yaml"))


def _tune_wandb(split_index: int, **kwargs):
    spec = ModelSpec("ElasticNet", from_recipe("scaledGeneExpression:fingerprints:elasticNet"))
    return _tune(spec, 2, fold_index=split_index, **kwargs)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    return install_fake_wandb(monkeypatch)


@pytest.mark.slow
def test_each_trial_is_one_named_run_with_its_metrics_in_the_summary(fake_wandb: FakeWandb) -> None:
    _tune_wandb(split_index=3, wandb_project="proj", wandb_base_config={"dataset": "synthetic"})

    assert fake_wandb.run_names() == ["ElasticNet_split3_trial0", "ElasticNet_split3_trial1"]
    for run in fake_wandb.runs:
        assert run.project == "proj"
        assert run.finished
        assert run.exit_code is None
        assert run.config["dataset"] == "synthetic"
        assert set(run.config) - {"dataset"} == set(ModelSpec.from_name("ElasticNet").hyperparameter_space)
        assert run.summary["hpo_metric"] == run.summary["val_RMSE"]
        assert {"val_R^2", "val_Pearson", "val_RMSE"} <= set(run.summary)


@pytest.mark.slow
def test_without_a_project_no_run_is_started(fake_wandb: FakeWandb) -> None:
    _tune_wandb(split_index=0)

    assert fake_wandb.runs == []


@pytest.mark.slow
def test_a_failing_trial_leaves_a_failed_run(fake_wandb: FakeWandb, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(DRPModel, "train", explode)

    with pytest.raises(HPOTrialsFailedError):
        _tune_wandb(split_index=0, wandb_project="proj")

    assert [run.exit_code for run in fake_wandb.runs] == [1, 1]


@pytest.mark.slow
def test_a_missing_wandb_fails_before_the_first_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    block_imports(monkeypatch, "wandb")

    with pytest.raises(ImportError, match="pip install wandb"):
        _tune_wandb(split_index=0, wandb_project="proj")

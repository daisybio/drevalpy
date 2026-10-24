"""HPO: search configuration, the trial loop and its runtime, and W&B tracking of trials."""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drevalpy.errors import DrEvalError
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.tuning import hpo
from drevalpy.models.tuning.config import HPOConfig, build_experiment_hpo_config, validate_hpo_metric
from drevalpy.models.tuning.hpo import HPOTrialsFailedError, hpam_tune
from drevalpy.models.tuning.hpo_runtime import _construct_trial_model, _mu_evaluate_trial_all_metrics
from drevalpy.types import SplitMask
from tests._fake_wandb import FakeWandb, install_fake_wandb
from tests._import_shims import block_imports
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
    synthetic_mudataset_identity,
)

SRC = Path(__file__).resolve().parents[3] / "src" / "drevalpy"


class TestValidateHpoMetric:
    @pytest.mark.parametrize("metric", sorted(AVAILABLE_METRICS), ids=lambda metric: metric.replace("^", ""))
    def test_accepts_every_available_metric(self, metric: str) -> None:
        assert validate_hpo_metric(metric) is None

    @pytest.mark.parametrize("metric", ["NotAMetric", "rmse"], ids=["unknown", "case-sensitive"])
    def test_rejects_an_invalid_metric_and_lists_the_choices(self, metric: str) -> None:
        with pytest.raises(ValueError, match=f"Invalid HPO metric '{metric}'.*RMSE"):
            validate_hpo_metric(metric)


class TestHPOConfig:
    def test_defaults_and_overrides(self) -> None:
        config = HPOConfig()
        assert (config.n_trials, config.metric, config.mode, config.random_state) == (16, "RMSE", "min", 42)
        assert dataclasses.is_dataclass(HPOConfig)

        config = HPOConfig(n_trials=3, metric="Pearson", mode="max", random_state=7)
        assert (config.n_trials, config.metric, config.mode, config.random_state) == (3, "Pearson", "max", 7)

    @pytest.mark.parametrize(
        ("metric", "expected_mode"),
        [("RMSE", "min"), ("MSE", "min"), ("MAE", "min"), ("Pearson", "max"), ("Spearman", "max"), ("R^2", "max")],
    )
    def test_from_metric_infers_the_mode(self, metric: str, expected_mode: str) -> None:
        config = HPOConfig.from_metric(metric)
        assert (config.mode, config.metric, config.n_trials) == (expected_mode, metric, 16)

    def test_from_metric_forwards_overrides(self) -> None:
        config = HPOConfig.from_metric("RMSE", n_trials=0, random_state=7)

        assert (config.n_trials, config.random_state) == (0, 7)
        with pytest.raises(TypeError):
            HPOConfig.from_metric("RMSE", not_a_field=1)

    def test_from_metric_validates_the_metric_before_n_trials(self) -> None:
        with pytest.raises(ValueError, match=r"n_trials must be >= 0 \(got -1\)"):
            HPOConfig.from_metric("RMSE", n_trials=-1)
        with pytest.raises(ValueError, match="Invalid HPO metric"):
            HPOConfig.from_metric("NotAMetric", n_trials=-1)


class TestBuildExperimentHpoConfig:
    def test_builds_from_the_metric(self) -> None:
        assert build_experiment_hpo_config("Pearson").mode == "max"
        config = build_experiment_hpo_config("MAE", n_trials=4, random_state=11)
        assert (config.metric, config.n_trials, config.random_state) == ("MAE", 4, 11)
        assert build_experiment_hpo_config("RMSE") == HPOConfig()

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="Invalid HPO metric"):
            build_experiment_hpo_config("NotAMetric")
        with pytest.raises(ValueError, match="n_trials must be >= 0"):
            build_experiment_hpo_config("RMSE", n_trials=-1)


def _tune(spec: ModelSpec, n_trials: int, **kwargs):
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=dataset.response_matrix.shape)
    return hpam_tune(
        model_spec=spec,
        mudataset=dataset,
        train_scope=scope,
        val_scope=scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=n_trials),
        **kwargs,
    )


def test_zero_trials_returns_spec_defaults() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    best, trials = _tune(spec, 0)

    assert best == spec.default_hyperparameters
    assert trials == []


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics", return_value=({"RMSE": 0.1}, np.zeros(4)))
def test_tuning_passes_model_spec_and_n_jobs_to_each_trial(mock_evaluate) -> None:
    best, _ = _tune(ModelSpec.from_name("ElasticNet"), 2, n_jobs=4)

    assert "alpha" in best
    assert mock_evaluate.call_count == 2
    assert [call.kwargs["n_jobs"] for call in mock_evaluate.call_args_list] == [4, 4]


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics", side_effect=RuntimeError("boom"))
def test_all_failed_trials_raise_without_default_fallback(mock_evaluate) -> None:
    with pytest.raises(HPOTrialsFailedError, match="All 2 hyperparameter trials failed"):
        _tune(ModelSpec.from_name("ElasticNet"), 2)
    assert issubclass(HPOTrialsFailedError, DrEvalError)
    assert issubclass(HPOTrialsFailedError, RuntimeError)


@patch(
    "drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics",
    side_effect=[({"RMSE": 1.0 - 0.1 * i}, np.zeros(4)) for i in range(4)],
)
def test_best_hyperparameters_carry_the_decoded_pow2_value_the_trials_trained_with(mock_evaluate) -> None:
    spec = ModelSpec("FingerprintElasticNet", from_spec("pca[gene_expression]:fingerprints:elasticNet"))

    best, trials = _tune(spec, 4)

    best_trial_params, _, _ = trials[-1]
    assert best["n_bits"] == best_trial_params["drug_featurizer.fingerprints.n_bits"]
    assert best["n_bits"] in {512, 1024, 2048}
    assert spec.resolve(best).values["drug_featurizer.fingerprints.n_bits"] == best["n_bits"]


def test_removed_legacy_hpo_artifacts_stay_removed() -> None:
    from drevalpy.models import drp_model
    from drevalpy.models.tuning.hpo import hpam_tune

    assert "ParameterGrid" not in inspect.getsource(hpam_tune)
    assert "yaml" not in inspect.getsource(drp_model.DRPModel.__init__).lower()
    literature = SRC / "components" / "predictors" / "literature"
    assert not (literature / "impl").exists()
    assert not list(literature.glob("*_predictor.py"))
    assert not list(SRC.rglob("hyperparameters.yaml"))


def _evaluate(predictions, **kwargs):
    model = MagicMock()
    model.predict.return_value = predictions
    scopes = lso_split_masks()
    call = {
        "mudataset": synthetic_mudataset_identity(),
        "train_scope": scopes.train,
        "val_scope": scopes.test,
        "early_stopping_scope": None,
        "model_checkpoint_dir": None,
    }
    call.update(kwargs)
    return model, _mu_evaluate_trial_all_metrics(model, **call)


def test_construct_trial_model_keeps_spec_identity() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    trial_model = _construct_trial_model(spec, {"predictor.elasticNet.alpha": 0.5})

    assert trial_model.spec.name == spec.name
    assert trial_model.hyperparameters["alpha"] == 0.5


@pytest.mark.parametrize("supplied", [False, True])
def test_trial_uses_one_training_call_and_keeps_nan_scope_alignment(tmp_path, supplied) -> None:
    dataset = synthetic_mudataset_identity()
    model, (metrics, predictions) = _evaluate(
        np.array([np.nan, 5.0]),
        mudataset=dataset,
        model_checkpoint_dir=tmp_path if supplied else None,
        trial_number=3,
    )

    model.train.assert_called_once()
    assert model.train.call_args.args == (dataset, lso_split_masks().train)
    assert model.train.call_args.kwargs["early_stopping_scope"] is None
    if supplied:
        assert model.train.call_args.kwargs["model_checkpoint_dir"] == tmp_path / "trial_3"
    assert metrics["MSE"] == 1.0
    assert np.isnan(metrics["R^2"])
    np.testing.assert_array_equal(predictions, [np.nan, 5.0])


def test_trial_rejects_prediction_truncation() -> None:
    with pytest.raises(ValueError, match="shape"):
        _evaluate(np.array([1.0]))


def test_trial_with_no_joint_valid_pairs_keeps_empty_metric_dictionary() -> None:
    _, (metrics, predictions) = _evaluate(np.full(2, np.nan))

    assert metrics == {}
    assert predictions.shape == (2,)


def test_trial_hands_n_jobs_to_the_training_call() -> None:
    model, _ = _evaluate(np.array([1.0, 2.0]), n_jobs=3)
    assert model.train.call_args.kwargs["n_jobs"] == 3
    model, _ = _evaluate(np.array([1.0, 2.0]))
    assert model.train.call_args.kwargs["n_jobs"] == 1


def _tune_wandb(**kwargs):
    spec = ModelSpec("ElasticNet", from_spec("scaledGeneExpression:fingerprints:elasticNet"))
    return _tune(spec, 2, **kwargs)


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
        assert len(run.config) > 1
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

    monkeypatch.setattr(hpo, "_mu_evaluate_trial_all_metrics", explode)

    with pytest.raises(HPOTrialsFailedError):
        _tune_wandb(split_index=0, wandb_project="proj")

    assert [run.exit_code for run in fake_wandb.runs] == [1, 1]


@pytest.mark.slow
def test_a_missing_wandb_fails_before_the_first_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    block_imports(monkeypatch, "wandb")

    with pytest.raises(ImportError, match="pip install wandb"):
        _tune_wandb(split_index=0, wandb_project="proj")

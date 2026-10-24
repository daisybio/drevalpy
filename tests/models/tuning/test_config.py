"""Tests for the Optuna search configuration in ``drevalpy.models.tuning.config``."""

from __future__ import annotations

import dataclasses

import pytest

from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.models.tuning.config import HPOConfig, build_experiment_hpo_config, validate_hpo_metric


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

"""Tests for the Optuna search configuration in ``drevalpy.models.tuning.config``."""

from __future__ import annotations

import dataclasses

import pytest

from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.models.tuning.config import (
    HPOConfig,
    build_experiment_hpo_config,
    validate_hpo_metric,
)


class TestValidateHpoMetric:
    """Metric names are checked against the evaluation registry."""

    @pytest.mark.parametrize("metric", sorted(AVAILABLE_METRICS), ids=lambda metric: metric.replace("^", ""))
    def test_accepts_every_available_metric(self, metric: str) -> None:
        assert validate_hpo_metric(metric) is None

    def test_rejects_an_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="Invalid HPO metric 'NotAMetric'"):
            validate_hpo_metric("NotAMetric")

    def test_error_lists_the_valid_choices(self) -> None:
        with pytest.raises(ValueError, match="RMSE"):
            validate_hpo_metric("NotAMetric")

    def test_metric_names_are_case_sensitive(self) -> None:
        with pytest.raises(ValueError, match="Invalid HPO metric 'rmse'"):
            validate_hpo_metric("rmse")


class TestHPOConfigDefaults:
    """The dataclass defaults are the documented experiment defaults."""

    def test_all_fields_default(self) -> None:
        config = HPOConfig()

        assert (config.n_trials, config.metric, config.mode, config.random_state) == (16, "RMSE", "min", 42)

    def test_fields_are_overridable(self) -> None:
        config = HPOConfig(n_trials=3, metric="Pearson", mode="max", random_state=7)

        assert (config.n_trials, config.metric, config.mode, config.random_state) == (3, "Pearson", "max", 7)

    def test_is_a_mutable_dataclass(self) -> None:
        assert dataclasses.is_dataclass(HPOConfig)
        assert [field.name for field in dataclasses.fields(HPOConfig)] == [
            "n_trials",
            "metric",
            "mode",
            "random_state",
        ]


class TestHPOConfigFromMetric:
    """``from_metric`` infers the optimization direction."""

    @pytest.mark.parametrize(
        ("metric", "expected_mode"),
        [
            pytest.param("RMSE", "min", id="rmse-minimized"),
            pytest.param("MSE", "min", id="mse-minimized"),
            pytest.param("MAE", "min", id="mae-minimized"),
            pytest.param("Pearson", "max", id="pearson-maximized"),
            pytest.param("Spearman", "max", id="spearman-maximized"),
            pytest.param("R^2", "max", id="r2-maximized"),
        ],
    )
    def test_infers_the_mode(self, metric: str, expected_mode: str) -> None:
        assert HPOConfig.from_metric(metric).mode == expected_mode

    def test_records_the_metric(self) -> None:
        assert HPOConfig.from_metric("Kendall").metric == "Kendall"

    def test_defaults_n_trials(self) -> None:
        assert HPOConfig.from_metric("RMSE").n_trials == 16

    def test_forwards_n_trials(self) -> None:
        assert HPOConfig.from_metric("RMSE", n_trials=5).n_trials == 5

    def test_accepts_zero_trials_for_default_only_tuning(self) -> None:
        assert HPOConfig.from_metric("RMSE", n_trials=0).n_trials == 0

    def test_rejects_negative_n_trials(self) -> None:
        with pytest.raises(ValueError, match=r"n_trials must be >= 0 \(got -1\)"):
            HPOConfig.from_metric("RMSE", n_trials=-1)

    def test_rejects_an_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="Invalid HPO metric"):
            HPOConfig.from_metric("NotAMetric")

    def test_validates_the_metric_before_n_trials(self) -> None:
        """A bad metric is reported even when ``n_trials`` is also invalid."""
        with pytest.raises(ValueError, match="Invalid HPO metric"):
            HPOConfig.from_metric("NotAMetric", n_trials=-1)

    def test_forwards_extra_field_overrides(self) -> None:
        assert HPOConfig.from_metric("RMSE", random_state=7).random_state == 7

    def test_rejects_an_unknown_field_override(self) -> None:
        with pytest.raises(TypeError):
            HPOConfig.from_metric("RMSE", not_a_field=1)


class TestBuildExperimentHpoConfig:
    """The shared entry point used for CV and final-model tuning."""

    def test_infers_the_mode_from_the_metric(self) -> None:
        assert build_experiment_hpo_config("Pearson").mode == "max"

    def test_carries_metric_trials_and_seed(self) -> None:
        config = build_experiment_hpo_config("MAE", n_trials=4, random_state=11)

        assert (config.metric, config.n_trials, config.random_state) == ("MAE", 4, 11)

    def test_defaults_match_the_dataclass_defaults(self) -> None:
        assert build_experiment_hpo_config("RMSE") == HPOConfig()

    def test_rejects_an_unknown_metric(self) -> None:
        with pytest.raises(ValueError, match="Invalid HPO metric"):
            build_experiment_hpo_config("NotAMetric")

    def test_rejects_negative_n_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials must be >= 0"):
            build_experiment_hpo_config("RMSE", n_trials=-1)

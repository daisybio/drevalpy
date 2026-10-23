"""Tests for :mod:`drevalpy.cli._options`: the option definitions several commands share.

The options are exercised on a probe command that declares every shared alias once, so the
parsing rules (defaults, case-insensitive choices, rejected values) are pinned here and the
command tests only check that each command forwards what it parsed.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
import typer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.cli._options import (
    GroupByOption,
    Hpo,
    HpoMetric,
    HpoNumTrials,
    HpoRandomState,
    RandomizationMode,
    ResponseTransformation,
    ResponseTransformationOption,
    WandbProject,
    resolve_response_transformation,
)
from drevalpy.cli._summary import GroupBy
from drevalpy.utils.response_transform import GroupMeanCenterer
from tests.cli._helpers import make_runner

runner = make_runner()


def _invoke_probe(*args: str) -> tuple[Any, dict[str, Any]]:
    """Run a command that declares every shared option and return its result and parsed values."""
    seen: dict[str, Any] = {}
    probe = typer.Typer()

    @probe.command()
    def cmd(
        hpo: Hpo = True,
        hpo_metric: HpoMetric = "RMSE",
        hpo_num_trials: HpoNumTrials = 16,
        hpo_random_state: HpoRandomState = None,
        response_transformation: ResponseTransformationOption = ResponseTransformation.NONE,
        group_by: GroupByOption = None,
        wandb_project: WandbProject = None,
        modes: Annotated[list[RandomizationMode] | None, typer.Option("--mode", case_sensitive=False)] = None,
    ) -> None:
        seen.update(
            hpo=hpo,
            hpo_metric=hpo_metric,
            hpo_num_trials=hpo_num_trials,
            hpo_random_state=hpo_random_state,
            response_transformation=response_transformation,
            group_by=group_by,
            wandb_project=wandb_project,
            modes=modes,
        )

    return runner.invoke(probe, list(args)), seen


class TestDefaults:
    def test_the_defaults_are_the_documented_ones(self) -> None:
        result, seen = _invoke_probe()

        assert result.exit_code == 0, result.output
        assert seen == {
            "hpo": True,
            "hpo_metric": "RMSE",
            "hpo_num_trials": 16,
            "hpo_random_state": None,
            "response_transformation": ResponseTransformation.NONE,
            "group_by": None,
            "wandb_project": None,
            "modes": None,
        }


class TestHpoAndTracking:
    def test_the_hpo_switch_has_a_negative_form(self) -> None:
        _, seen = _invoke_probe("--no-hpo")

        assert seen["hpo"] is False

    def test_the_tuning_options_are_parsed(self) -> None:
        _, seen = _invoke_probe("--hpo-metric", "Pearson", "--hpo-trials", "3", "--hpo-random-state", "7")

        assert (seen["hpo_metric"], seen["hpo_num_trials"], seen["hpo_random_state"]) == ("Pearson", 3, 7)

    def test_non_integer_trials_are_a_usage_error(self) -> None:
        result, _ = _invoke_probe("--hpo-trials", "many")

        assert result.exit_code == 2

    def test_the_wandb_project_is_parsed(self) -> None:
        _, seen = _invoke_probe("--wandb-project", "drevalpy")

        assert seen["wandb_project"] == "drevalpy"


class TestGroupBy:
    @pytest.mark.parametrize(("option", "expected"), [("drug", GroupBy.DRUG), ("SAMPLE", GroupBy.SAMPLE)])
    def test_the_choice_is_case_insensitive(self, option: str, expected: GroupBy) -> None:
        _, seen = _invoke_probe("--group-by", option)

        assert seen["group_by"] is expected

    def test_an_unknown_grouping_is_a_usage_error(self) -> None:
        result, _ = _invoke_probe("--group-by", "tissue")

        assert result.exit_code == 2


class TestRandomizationMode:
    def test_repeated_modes_are_collected_in_order(self) -> None:
        _, seen = _invoke_probe("--mode", "SVRC", "--mode", "svcd")

        assert seen["modes"] == [RandomizationMode.SVRC, RandomizationMode.SVCD]

    def test_an_unknown_mode_is_a_usage_error(self) -> None:
        result, _ = _invoke_probe("--mode", "SVXX")

        assert result.exit_code == 2

    def test_the_members_are_the_four_documented_codes(self) -> None:
        assert [mode.value for mode in RandomizationMode] == ["SVRC", "SVCC", "SVRD", "SVCD"]


class TestResponseTransformation:
    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            pytest.param("standard", StandardScaler, id="standard"),
            pytest.param("minmax", MinMaxScaler, id="minmax"),
            pytest.param("robust", RobustScaler, id="robust"),
            pytest.param("drug_mean", GroupMeanCenterer, id="drug_mean"),
            pytest.param("drug_tissue_mean", GroupMeanCenterer, id="drug_tissue_mean"),
        ],
    )
    def test_each_choice_resolves_to_its_transformer(self, option: str, expected: type) -> None:
        _, seen = _invoke_probe("--response-transformation", option)

        assert isinstance(resolve_response_transformation(seen["response_transformation"]), expected)

    @pytest.mark.parametrize("option", ["none", "None", "NONE"])
    def test_none_resolves_to_no_transformer(self, option: str) -> None:
        _, seen = _invoke_probe("--response-transformation", option)

        assert resolve_response_transformation(seen["response_transformation"]) is None

    def test_the_default_resolves_to_no_transformer(self) -> None:
        assert resolve_response_transformation(ResponseTransformation.NONE) is None

    def test_the_transformer_is_handed_over_unfitted(self) -> None:
        transformer = resolve_response_transformation(ResponseTransformation.STANDARD)

        assert not hasattr(transformer, "mean_")

    @pytest.mark.parametrize(
        ("choice", "levels"),
        [
            pytest.param(ResponseTransformation.DRUG_MEAN, ("drug",), id="drug_mean"),
            pytest.param(ResponseTransformation.DRUG_TISSUE_MEAN, ("drug", "tissue"), id="drug_tissue_mean"),
        ],
    )
    def test_the_group_means_centre_on_their_levels(
        self, choice: ResponseTransformation, levels: tuple[str, ...]
    ) -> None:
        assert resolve_response_transformation(choice).levels == levels

    def test_each_call_builds_a_new_transformer(self) -> None:
        first = resolve_response_transformation(ResponseTransformation.MINMAX)

        assert first is not resolve_response_transformation(ResponseTransformation.MINMAX)

    def test_an_unknown_choice_is_a_usage_error(self) -> None:
        result, _ = _invoke_probe("--response-transformation", "logarithmic")

        assert result.exit_code == 2

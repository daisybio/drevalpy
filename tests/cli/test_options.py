"""Tests for :mod:`drevalpy.cli._options`: the option definitions several commands share.

The options are exercised on a probe command that declares every shared alias once, so the
parsing rules are pinned here and the command tests only check that each command forwards what it parsed.
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


def test_the_defaults_are_the_documented_ones() -> None:
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


def test_the_tuning_and_tracking_options_are_parsed() -> None:
    _, seen = _invoke_probe(
        "--no-hpo", "--hpo-metric", "Pearson", "--hpo-trials", "3", "--hpo-random-state", "7", "--wandb-project", "d"
    )

    assert seen["hpo"] is False
    assert (seen["hpo_metric"], seen["hpo_num_trials"], seen["hpo_random_state"]) == ("Pearson", 3, 7)
    assert seen["wandb_project"] == "d"


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["--hpo-trials", "many"], id="non-integer-trials"),
        pytest.param(["--group-by", "tissue"], id="unknown-grouping"),
        pytest.param(["--mode", "SVXX"], id="unknown-mode"),
        pytest.param(["--response-transformation", "logarithmic"], id="unknown-transformation"),
    ],
)
def test_bad_values_are_usage_errors(args: list[str]) -> None:
    result, _ = _invoke_probe(*args)

    assert result.exit_code == 2


def test_group_by_and_randomization_modes_are_case_insensitive() -> None:
    _, seen = _invoke_probe("--group-by", "SAMPLE", "--mode", "SVRC", "--mode", "svcd")

    assert seen["group_by"] is GroupBy.SAMPLE
    assert seen["modes"] == [RandomizationMode.SVRC, RandomizationMode.SVCD]
    assert [mode.value for mode in RandomizationMode] == ["SVRC", "SVCC", "SVRD", "SVCD"]


class TestResponseTransformation:
    @pytest.mark.parametrize(
        ("option", "expected"),
        [
            ("standard", StandardScaler),
            ("minmax", MinMaxScaler),
            ("robust", RobustScaler),
            ("drug_mean", GroupMeanCenterer),
            ("drug_tissue_mean", GroupMeanCenterer),
        ],
    )
    def test_each_choice_resolves_to_a_new_unfitted_transformer(self, option: str, expected: type) -> None:
        _, seen = _invoke_probe("--response-transformation", option)

        transformer = resolve_response_transformation(seen["response_transformation"])

        assert isinstance(transformer, expected)
        assert not hasattr(transformer, "mean_")
        assert transformer is not resolve_response_transformation(seen["response_transformation"])

    @pytest.mark.parametrize("option", ["none", "NONE", None])
    def test_none_and_the_default_resolve_to_no_transformer(self, option: str | None) -> None:
        _, seen = _invoke_probe(*(["--response-transformation", option] if option else []))

        assert resolve_response_transformation(seen["response_transformation"]) is None

    @pytest.mark.parametrize(
        ("choice", "levels"),
        [(ResponseTransformation.DRUG_MEAN, ("drug",)), (ResponseTransformation.DRUG_TISSUE_MEAN, ("drug", "tissue"))],
    )
    def test_the_group_means_centre_on_their_levels(
        self, choice: ResponseTransformation, levels: tuple[str, ...]
    ) -> None:
        assert resolve_response_transformation(choice).levels == levels

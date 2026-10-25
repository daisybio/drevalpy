"""Tests for the option definitions the commands share (:mod:`drevalpy.cli._options`, `_quality`, `_summary`)."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
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
from drevalpy.cli._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli
from drevalpy.cli._summary import format_grouped_summary
from drevalpy.data.quality import CurveQuality
from drevalpy.evaluation import Metric, evaluate
from drevalpy.models.response_transformation import GroupMeanCenterer
from drevalpy.types.results import ExperimentResult
from tests.cli._helpers import make_runner
from tests.synthetic import make_run_result

runner = make_runner()


def _invoke_probe(*args: str) -> tuple[Any, dict[str, Any]]:
    seen: dict[str, Any] = {}
    probe = typer.Typer()

    @probe.command()
    def cmd(
        hpo: Hpo = True,
        hpo_metric: HpoMetric = Metric.RMSE,
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
        pytest.param(["--mode", "SVXX"], id="unknown-mode"),
        pytest.param(["--response-transformation", "logarithmic"], id="unknown-transformation"),
    ],
)
def test_bad_values_are_usage_errors(args: list[str]) -> None:
    result, _ = _invoke_probe(*args)

    assert result.exit_code == 2


def test_group_by_and_randomization_modes_are_case_insensitive() -> None:
    _, seen = _invoke_probe("--group-by", "SAMPLE", "--mode", "SVRC", "--mode", "svcd")

    assert seen["group_by"] == "sample"
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


def test_quality_from_cli_rules() -> None:
    default = CurveQuality.default()
    assert quality_from_cli(False, None, None) == default
    assert quality_from_cli(True, None, None) == CurveQuality.off()
    assert quality_from_cli(False, 3.0, None) == CurveQuality(3.0, default.min_abs_fold_change)
    assert quality_from_cli(False, 2.0, 0.7) == CurveQuality(min_relevance_score=2.0, min_abs_fold_change=0.7)
    assert quality_from_cli(False, 0.0, None).min_relevance_score == 0.0


@pytest.mark.parametrize(
    ("relevance", "fold_change", "flag"),
    [(1.0, None, "--min-relevance-score"), (None, 0.5, "--min-abs-fold-change")],
)
def test_no_filter_with_a_threshold_is_a_bad_parameter(
    relevance: float | None, fold_change: float | None, flag: str
) -> None:
    with pytest.raises(typer.BadParameter, match=f"{flag}.*--no-quality-filter"):
        quality_from_cli(True, relevance, fold_change)


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def show(
        no_quality_filter: NoQualityFilter = False,
        min_relevance_score: MinRelevanceScore = None,
        min_abs_fold_change: MinAbsFoldChange = None,
    ) -> None:
        typer.echo(quality_from_cli(no_quality_filter, min_relevance_score, min_abs_fold_change).to_metadata())

    return app


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], str(CurveQuality.default().to_metadata())),
        (
            ["--min-relevance-score", "2.5", "--min-abs-fold-change", "0.1"],
            str({"min_relevance_score": 2.5, "min_abs_fold_change": 0.1}),
        ),
        (["--no-quality-filter"], "{}"),
    ],
    ids=["defaults", "thresholds", "off"],
)
def test_options(argv: list[str], expected: str) -> None:
    result = runner.invoke(_app(), argv)

    assert result.exit_code == 0, result.output
    assert result.output.strip() == expected


def test_combining_no_filter_with_a_threshold_is_a_usage_error() -> None:
    assert runner.invoke(_app(), ["--no-quality-filter", "--min-relevance-score", "2"]).exit_code == 2


def _experiment() -> ExperimentResult:
    constant = make_run_result(model_name="DrugMean", n_pairs=40)
    constant.predictions = np.array([float(i % 4) for i in range(40)])
    constant.metrics = evaluate(constant.predictions, constant.ground_truth, ["MSE", "Pearson"])
    varying = make_run_result(model_name="Varying", n_pairs=40)
    varying.metrics = evaluate(varying.predictions, varying.ground_truth, ["MSE", "Pearson"])
    return ExperimentResult([constant, varying])


def test_summary_layout_shows_group_counts_and_scores_a_constant_prediction_zero() -> None:
    lines = format_grouped_summary(_experiment(), "drug").splitlines()

    assert format_grouped_summary(_experiment(), "sample").startswith("Metrics computed within each sample")
    assert lines[1].split() == ["Model", "MSE", "Pearson"]
    assert [line.split()[0] for line in lines[2:]] == ["DrugMean", "Varying"]
    constant = next(line for line in lines if line.startswith("DrugMean"))
    assert constant.split()[-2:] == ["0.000", "(4)"]
    assert "undefined" not in constant
    assert "nan" not in constant.lower()
    assert "(4)" in next(line for line in lines if line.startswith("Varying"))


def test_summary_layout_labels_undefined_metrics_and_shows_group_counts() -> None:
    single_pair_per_drug = make_run_result(model_name="Single", n_pairs=4, n_drugs=4)
    single_pair_per_drug.metrics = evaluate(
        single_pair_per_drug.predictions, single_pair_per_drug.ground_truth, ["MSE", "Pearson"]
    )

    lines = format_grouped_summary(ExperimentResult([single_pair_per_drug]), "drug").splitlines()

    row = lines[2].split()
    assert row[0] == "Single"
    assert row[2] == "(4)"
    assert row[3:] == ["undefined"]
    assert "nan" not in lines[2].lower()


def _prefixed_experiment() -> ExperimentResult:
    run = make_run_result(model_name="Varying", n_pairs=40, n_samples=8, n_drugs=5)
    run.sample_ids = np.array([f"{'GDSC2' if i % 2 else 'CTRPv2'}::CL_{i % 8}" for i in range(40)])
    run.metrics = evaluate(run.predictions, run.ground_truth, ["MSE", "Pearson"])
    return ExperimentResult([run])


def test_grouping_by_study_scores_each_prefix_of_a_concatenated_run() -> None:
    lines = format_grouped_summary(_prefixed_experiment(), "study").splitlines()

    assert lines[0].startswith("Metrics computed within each study, averaged over studies and folds")
    assert "(2)" in next(line for line in lines if line.startswith("Varying"))


def test_grouping_by_study_on_a_plain_run_is_undefined() -> None:
    lines = format_grouped_summary(_experiment(), "study").splitlines()

    assert all("undefined" in line for line in lines[2:])

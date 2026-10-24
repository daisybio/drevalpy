"""Tests for :mod:`drevalpy.cli._quality`, the curve-quality options shared by ``run`` and ``data split``."""

from __future__ import annotations

import pytest
import typer

from drevalpy.cli._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli
from drevalpy.data.quality import CurveQuality
from tests.cli._helpers import make_runner

runner = make_runner()


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

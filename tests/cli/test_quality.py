"""Tests for :mod:`drevalpy.cli._quality`, the curve-quality options shared by ``run`` and ``data split``."""

from __future__ import annotations

import pytest
import typer

from drevalpy.cli._quality import MinAbsFoldChange, MinRelevanceScore, NoQualityFilter, quality_from_cli
from drevalpy.data.quality import CurveQuality
from tests.cli._helpers import make_runner

runner = make_runner()


class TestQualityFromCli:
    def test_no_options_give_the_default_rule(self) -> None:
        assert quality_from_cli(False, None, None) == CurveQuality.default()

    def test_no_filter_gives_the_off_rule(self) -> None:
        assert quality_from_cli(True, None, None) == CurveQuality.off()

    def test_a_given_threshold_overrides_the_default_one(self) -> None:
        rule = quality_from_cli(False, 3.0, None)

        assert rule.min_relevance_score == 3.0
        assert rule.min_abs_fold_change == CurveQuality.default().min_abs_fold_change

    def test_both_thresholds_can_be_overridden(self) -> None:
        assert quality_from_cli(False, 2.0, 0.7) == CurveQuality(min_relevance_score=2.0, min_abs_fold_change=0.7)

    def test_a_zero_threshold_is_a_given_threshold(self) -> None:
        assert quality_from_cli(False, 0.0, None).min_relevance_score == 0.0

    @pytest.mark.parametrize(
        ("relevance", "fold_change", "flag"),
        [
            pytest.param(1.0, None, "--min-relevance-score", id="relevance"),
            pytest.param(None, 0.5, "--min-abs-fold-change", id="fold-change"),
        ],
    )
    def test_no_filter_with_a_threshold_is_a_bad_parameter(
        self, relevance: float | None, fold_change: float | None, flag: str
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


class TestOptions:
    def test_defaults(self) -> None:
        result = runner.invoke(_app(), [])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == str(CurveQuality.default().to_metadata())

    def test_thresholds_are_parsed_as_floats(self) -> None:
        result = runner.invoke(_app(), ["--min-relevance-score", "2.5", "--min-abs-fold-change", "0.1"])

        assert result.output.strip() == str({"min_relevance_score": 2.5, "min_abs_fold_change": 0.1})

    def test_no_quality_filter(self) -> None:
        result = runner.invoke(_app(), ["--no-quality-filter"])

        assert result.output.strip() == "{}"

    def test_combining_them_is_a_usage_error(self) -> None:
        result = runner.invoke(_app(), ["--no-quality-filter", "--min-relevance-score", "2"])

        assert result.exit_code == 2

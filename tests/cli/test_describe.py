"""Tests for the ``describe`` command."""

from __future__ import annotations

import csv
import io

from upath import UPath

from drevalpy.cli.describe import COLUMNS
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

HEADER = [
    "model",
    "predictor",
    "sample_featurizer",
    "drug_featurizer",
    "accelerator",
    "early_stopping",
    "licenses",
]


def _parse(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def test_help_lists_the_options() -> None:
    output = plain(runner.invoke(app, ["describe", "-h"], env=HELP_ENV).output)

    for option in ("--sample-featurizer", "--drug-featurizer", "--predictor", "--tsv", "Nextflow"):
        assert option in output


def test_the_table_lists_each_model() -> None:
    result = runner.invoke(app, ["describe", "Ridge", "DIPK"], env=HELP_ENV)

    assert result.exit_code == 0, result.output
    assert "Ridge" in result.stdout
    assert "DIPK" in result.stdout


def test_tsv_file_holds_a_header_and_one_row_per_model(tmp_path: UPath) -> None:
    out = tmp_path / "out.tsv"

    result = runner.invoke(app, ["describe", "Ridge", "DIPK", "--tsv", str(out)], env=HELP_ENV)

    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0].split("\t") == HEADER == list(COLUMNS)
    rows = {row["model"]: row for row in _parse(text)}
    assert list(rows) == ["Ridge", "DIPK"]
    assert rows["Ridge"] == {
        "model": "Ridge",
        "predictor": "ridge",
        "sample_featurizer": "scaledGeneExpression",
        "drug_featurizer": "fingerprints",
        "accelerator": "none",
        "early_stopping": "none",
        "licenses": "MIT",
    }
    assert rows["DIPK"]["predictor"] == "dipk"
    assert rows["DIPK"]["accelerator"] == "optional"


def test_tsv_to_stdout_is_clean_and_the_table_goes_to_stderr() -> None:
    result = runner.invoke(app, ["describe", "Ridge", "DIPK", "--tsv", "-"], env=HELP_ENV)

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0].split("\t") == HEADER
    assert [row["model"] for row in _parse(result.stdout)] == ["Ridge", "DIPK"]
    assert "Models" in result.stderr


def test_tsv_into_a_missing_directory_fails(tmp_path: UPath) -> None:
    result = runner.invoke(app, ["describe", "Ridge", "--tsv", str(tmp_path / "missing" / "out.tsv")])

    assert result.exit_code != 0


def test_a_grid_expands_and_skipped_notes_go_to_stderr() -> None:
    argv = ["describe", "-S", "pca[gene_expression]", "-D", "fingerprints", "-P", "ridge", "-P", "dipk", "--tsv", "-"]

    result = runner.invoke(app, argv, env=HELP_ENV)

    assert result.exit_code == 0, result.output
    rows = _parse(result.stdout)
    assert "pca[gene_expression]:fingerprints:ridge" in [row["model"] for row in rows]
    ridge = next(row for row in rows if row["predictor"] == "ridge")
    assert (ridge["sample_featurizer"], ridge["drug_featurizer"]) == ("pca[gene_expression]", "fingerprints")
    if len(rows) == 1:
        assert "Skipped" in result.stderr
        assert "Skipped" not in result.stdout


def test_an_unknown_model_is_an_error() -> None:
    result = runner.invoke(app, ["describe", "NotAModel"], env=HELP_ENV)

    assert result.exit_code == 1
    assert "Error:" in result.output


def test_no_model_and_no_predictor_is_a_usage_error() -> None:
    assert runner.invoke(app, ["describe"], env=HELP_ENV).exit_code == 2

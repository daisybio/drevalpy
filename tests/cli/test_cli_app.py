"""Tests for the Typer-based ``drevalpy`` CLI."""

from __future__ import annotations

import re
import warnings
from typing import cast

import pytest
from typer.testing import CliRunner

from drevalpy.cli._helpers import normalize_list_argv
from drevalpy.cli.legacy import load_response
from drevalpy.cli.main import app

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_stdout(text: str) -> str:
    """Strip terminal escape codes (e.g. when CI sets ``FORCE_COLOR=1``).

    :param text: Captured CLI stdout.
    :return: *text* without ANSI escape sequences.
    """
    return _ANSI_ESCAPE.sub("", text)


def test_drevalpy_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "viability-preprocess" in result.stdout
    assert "train-cv" in result.stdout
    assert "make-pipeline-report" in result.stdout


def test_viability_preprocess_help() -> None:
    result = runner.invoke(
        app,
        ["viability-preprocess", "--help"],
        env={"FORCE_COLOR": "1", "CI": "true"},
    )
    assert result.exit_code == 0
    help_text = _plain_stdout(result.stdout)
    assert "--dataset_name" in help_text
    assert "--path_data" in help_text
    assert "--cores" in help_text


def test_load_response_requires_response_dataset() -> None:
    result = runner.invoke(app, ["load-response"])
    assert result.exit_code != 0


def test_legacy_load_response_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy script warns and forwards argv to the Typer subcommand."""
    monkeypatch.setattr("sys.argv", ["drevalpy-load-response", "--help"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit) as exc_info:
            load_response()

    assert exc_info.value.code == 0
    assert any(issubclass(w.category, FutureWarning) and "drevalpy load-response" in str(w.message) for w in caught)


def test_pipeline_root_missing_models_fails_fast() -> None:
    result = runner.invoke(
        app,
        normalize_list_argv(
            [
                "--dataset_name",
                "GDSC1",
            ]
        ),
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "At least one model must be specified" in str(result.exception)


def test_pipeline_accepts_space_separated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_check(args: object) -> None:
        captured["models"] = list(args.models)  # type: ignore[attr-defined]

    def fake_main(args: object) -> None:
        return None

    monkeypatch.setattr("drevalpy.cli.pipeline.check_arguments", fake_check)
    monkeypatch.setattr("drevalpy.cli.pipeline.main", fake_main)

    result = runner.invoke(
        app,
        normalize_list_argv(["--models", "Simple", "ElasticNet", "--dataset_name", "GDSC1"]),
    )
    assert result.exit_code == 0
    assert captured["models"] == ["Simple", "ElasticNet"]


def test_evaluate_hpams_accepts_space_separated_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(**kwargs: object) -> None:
        captured["hpam_yamls"] = list(cast("list[str]", kwargs["hpam_yamls"]))
        captured["pred_datas"] = list(cast("list[str]", kwargs["pred_datas"]))

    monkeypatch.setattr("drevalpy.cli.evaluate_hpams.run_evaluate_and_find_max", fake_run)

    result = runner.invoke(
        app,
        normalize_list_argv(
            [
                "evaluate-hpams",
                "--model_name",
                "m",
                "--split_id",
                "0",
                "--hpam_yamls",
                "a.yml",
                "b.yml",
                "--pred_datas",
                "p1.pkl",
                "p2.pkl",
            ]
        ),
    )
    assert result.exit_code == 0
    assert captured["hpam_yamls"] == ["a.yml", "b.yml"]
    assert captured["pred_datas"] == ["p1.pkl", "p2.pkl"]


def test_dash_h_shows_root_help() -> None:
    """``-h`` is accepted as an alias for ``--help`` on the root command."""
    result = runner.invoke(app, ["-h"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_dash_h_shows_subcommand_help() -> None:
    """``-h`` propagates to subcommands (e.g. ``drevalpy report -h``)."""
    result = runner.invoke(app, ["report", "-h"])
    assert result.exit_code == 0
    help_text = _plain_stdout(result.stdout)
    assert "--dataset_name" in help_text


def test_report_uses_dataset_name_option() -> None:
    """``report`` exposes ``--dataset_name`` and no longer accepts ``--dataset``."""
    help_result = runner.invoke(app, ["report", "--help"], env={"FORCE_COLOR": "1", "CI": "true"})
    assert help_result.exit_code == 0
    assert "--dataset_name" in _plain_stdout(help_result.stdout)

    rejected = runner.invoke(app, ["report", "--run_id", "x", "--dataset", "TOYv1"])
    # Click reports unknown options with a usage error (exit code 2); assert on that
    # specifically so the test cannot pass via an unrelated downstream failure.
    assert rejected.exit_code == 2
    assert "No such option" in _plain_stdout(rejected.output)


def test_report_forwards_dataset_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``report --dataset_name`` is forwarded to ``run_report`` as ``dataset``."""
    captured: dict[str, object] = {}

    def fake_run_report(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("drevalpy.cli.report.run_report", fake_run_report)

    result = runner.invoke(app, ["report", "--run_id", "my_run", "--dataset_name", "TOYv1"])
    assert result.exit_code == 0
    assert captured["run_id"] == "my_run"
    assert captured["dataset"] == "TOYv1"


def test_pipeline_help_uses_valid_randomization_example() -> None:
    result = runner.invoke(
        app,
        ["--help"],
        env={"FORCE_COLOR": "1", "CI": "true"},
    )
    help_text = _plain_stdout(result.stdout)
    assert "SVCC" in help_text
    assert "SVCD" in help_text
    assert "SCVC" not in help_text
    assert "SCVD" not in help_text

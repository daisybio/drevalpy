"""Tests for the per-module coverage floor in ``tools/coverage_gate.py``."""

from __future__ import annotations

import json
from typing import Any

import pytest
from upath import UPath

from tools.coverage_gate import main as coverage_main


def _write_coverage_json(path: UPath, percentages: dict[str, float]) -> UPath:
    report: dict[str, Any] = {
        "files": {path_: {"summary": {"percent_covered": percent}} for path_, percent in percentages.items()}
    }
    target = path / "coverage.json"
    target.write_text(json.dumps(report))
    return target


def _write_coverage_pyproject(path: UPath, min_file_coverage: int, exemptions: dict[str, int]) -> UPath:
    lines = ["[tool.drevalpy.coverage_gate]", f"min_file_coverage = {min_file_coverage}"]
    if exemptions:
        lines.append("[tool.drevalpy.coverage_gate.exemptions]")
        lines.extend(f'"{key}" = {value}' for key, value in exemptions.items())
    target = path / "pyproject.toml"
    target.write_text("\n".join(lines))
    return target


def _run_coverage_gate(coverage_json: UPath, pyproject: UPath) -> int:
    return coverage_main(["--coverage-json", str(coverage_json), "--pyproject", str(pyproject)])


@pytest.mark.parametrize(
    ("path", "percent", "exemptions", "expected_exit_code"),
    [
        ("drevalpy/a.py", 50.0, {}, 0),
        ("drevalpy/a.py", 100.0, {}, 0),
        ("drevalpy/a.py", 49.999, {}, 1),
        ("drevalpy/a.py", 0.0, {}, 1),
        ("drevalpy/low.py", 20.0, {"drevalpy/low.py": 20}, 0),
        ("drevalpy/low.py", 15.0, {"drevalpy/low.py": 20}, 1),
        ("drevalpy\\low.py", 20.0, {"drevalpy/low.py": 20}, 0),
        ("src/drevalpy/low.py", 25.0, {"drevalpy/low.py": 20}, 0),
    ],
    ids=["at-floor", "full", "just-below", "never-imported", "exempt-ok", "exempt-below", "windows-sep", "src-layout"],
)
def test_the_floor_is_inclusive_and_exemptions_replace_it(
    tmp_path: UPath, path: str, percent: float, exemptions: dict[str, int], expected_exit_code: int
) -> None:
    coverage_json = _write_coverage_json(tmp_path, {path: percent})
    pyproject = _write_coverage_pyproject(tmp_path, min_file_coverage=50, exemptions=exemptions)

    assert _run_coverage_gate(coverage_json, pyproject) == expected_exit_code


def test_coverage_report_names_only_the_offending_module(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0, "drevalpy/b.py": 10.0})

    _run_coverage_gate(coverage_json, _write_coverage_pyproject(tmp_path, min_file_coverage=50, exemptions={}))

    out = capsys.readouterr().out
    assert "drevalpy/b.py" in out
    assert "drevalpy/a.py" not in out


def test_reports_exemptions_that_can_be_lowered_or_whose_module_is_absent(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/improved.py": 70.0})
    pyproject = _write_coverage_pyproject(
        tmp_path, min_file_coverage=50, exemptions={"drevalpy/improved.py": 20, "drevalpy/gone.py": 20}
    )

    assert _run_coverage_gate(coverage_json, pyproject) == 0

    out = capsys.readouterr().out
    assert "can be lowered or deleted" in out
    assert "drevalpy/gone.py" in out


@pytest.mark.parametrize("missing", ["coverage-json", "pyproject"])
def test_exits_with_a_clear_message_when_an_input_is_missing(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    coverage_json = tmp_path / "coverage.json" if missing == "coverage-json" else _write_coverage_json(tmp_path, {})
    pyproject = tmp_path / "absent.toml" if missing == "pyproject" else _write_coverage_pyproject(tmp_path, 50, {})

    with pytest.raises(SystemExit) as excinfo:
        _run_coverage_gate(coverage_json, pyproject)

    assert excinfo.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_defaults_the_floor_when_the_config_table_is_absent(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 49.0})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')

    assert _run_coverage_gate(coverage_json, pyproject) == 1

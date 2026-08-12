"""Tests for the per-module coverage floor gate in ``tools/coverage_gate.py``."""

from __future__ import annotations

import json
from typing import Any

import pytest
from upath import UPath

from tools.coverage_gate import main


def _write_coverage_json(path: UPath, percentages: dict[str, float]) -> UPath:
    """Write a minimal ``coverage.json`` holding the given per-file percentages."""
    report: dict[str, Any] = {
        "files": {path_: {"summary": {"percent_covered": percent}} for path_, percent in percentages.items()}
    }
    target = path / "coverage.json"
    target.write_text(json.dumps(report))
    return target


def _write_pyproject(path: UPath, min_file_coverage: int, exemptions: dict[str, int]) -> UPath:
    """Write a minimal ``pyproject.toml`` holding a coverage-gate config table."""
    lines = ["[tool.drevalpy.coverage_gate]", f"min_file_coverage = {min_file_coverage}"]
    if exemptions:
        lines.append("[tool.drevalpy.coverage_gate.exemptions]")
        lines.extend(f'"{key}" = {value}' for key, value in exemptions.items())
    target = path / "pyproject.toml"
    target.write_text("\n".join(lines))
    return target


def _run(coverage_json: UPath, pyproject: UPath) -> int:
    return main(["--coverage-json", str(coverage_json), "--pyproject", str(pyproject)])


def test_passes_when_every_module_meets_the_floor(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0, "drevalpy/b.py": 50.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={})

    assert _run(coverage_json, pyproject) == 0


def test_fails_when_a_module_is_below_the_floor(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0, "drevalpy/b.py": 49.9})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={})

    assert _run(coverage_json, pyproject) == 1


def test_names_the_offending_module_in_the_report(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0, "drevalpy/b.py": 10.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={})

    _run(coverage_json, pyproject)

    out = capsys.readouterr().out
    assert "drevalpy/b.py" in out
    assert "drevalpy/a.py" not in out


def test_exempted_module_below_global_floor_but_above_own_floor_passes(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/low.py": 20.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={"drevalpy/low.py": 20})

    assert _run(coverage_json, pyproject) == 0


def test_exempted_module_below_its_own_floor_fails(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/low.py": 15.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={"drevalpy/low.py": 20})

    assert _run(coverage_json, pyproject) == 1


def test_reports_exemptions_that_can_be_ratcheted_down(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/improved.py": 70.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={"drevalpy/improved.py": 20})

    exit_code = _run(coverage_json, pyproject)

    assert exit_code == 0
    assert "can be lowered or deleted" in capsys.readouterr().out


def test_reports_exemptions_for_modules_absent_from_the_report(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={"drevalpy/gone.py": 20})

    exit_code = _run(coverage_json, pyproject)

    assert exit_code == 0
    assert "drevalpy/gone.py" in capsys.readouterr().out


def test_normalizes_windows_separators_in_report_paths(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy\\low.py": 20.0})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={"drevalpy/low.py": 20})

    assert _run(coverage_json, pyproject) == 0


def test_exits_with_a_clear_message_when_the_coverage_json_is_missing(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={})
    missing = tmp_path / "coverage.json"

    with pytest.raises(SystemExit) as excinfo:
        _run(missing, pyproject)

    assert excinfo.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_exits_when_the_pyproject_is_missing(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 80.0})
    missing = tmp_path / "absent.toml"

    with pytest.raises(SystemExit) as excinfo:
        _run(coverage_json, missing)

    assert excinfo.value.code == 1
    assert "not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("percent", "expected_exit_code"),
    [
        pytest.param(50.0, 0, id="exactly-at-floor-passes"),
        pytest.param(49.999, 1, id="just-below-floor-fails"),
        pytest.param(0.0, 1, id="never-imported-module-fails"),
        pytest.param(100.0, 0, id="fully-covered-passes"),
    ],
)
def test_floor_is_inclusive(tmp_path: UPath, percent: float, expected_exit_code: int) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": percent})
    pyproject = _write_pyproject(tmp_path, min_file_coverage=50, exemptions={})

    assert _run(coverage_json, pyproject) == expected_exit_code


def test_defaults_the_floor_when_the_config_table_is_absent(tmp_path: UPath) -> None:
    coverage_json = _write_coverage_json(tmp_path, {"drevalpy/a.py": 49.0})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')

    assert _run(coverage_json, pyproject) == 1

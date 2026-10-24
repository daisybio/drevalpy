"""Tests for the repo gates in ``tools/``: the per-module coverage floor and the statement ceiling."""

from __future__ import annotations

import json
from typing import Any

import pytest
from upath import UPath

from tools.coverage_gate import main as coverage_main
from tools.size_gate import count_statements
from tools.size_gate import main as size_main


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


def _write_package(path: UPath, modules: dict[str, str]) -> UPath:
    package = path / "drevalpy"
    for relative, source in modules.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return package


def _statements(count: int) -> str:
    return "\n".join(f"x{index} = {index}" for index in range(count)) + "\n" if count else ""


def _write_size_pyproject(path: UPath, max_module_statements: int, exemptions: dict[str, int]) -> UPath:
    """Write a minimal ``pyproject.toml``; exemption keys are package-relative, the form the gate reports."""
    lines = ["[tool.drevalpy.size_gate]", f"max_module_statements = {max_module_statements}"]
    if exemptions:
        lines.append("[tool.drevalpy.size_gate.exemptions]")
        lines.extend(f'"drevalpy/{key}" = {value}' for key, value in exemptions.items())
    target = path / "pyproject.toml"
    target.write_text("\n".join(lines))
    return target


def _run_size_gate(package: UPath, pyproject: UPath) -> int:
    return size_main(["--package", str(package), "--pyproject", str(pyproject)])


@pytest.mark.parametrize(
    ("modules", "exemptions", "expected_exit_code"),
    [
        ({"a.py": 5, "b.py": 10}, {}, 0),
        ({"a.py": 0}, {}, 0),
        ({"a.py": 5, "b.py": 11}, {}, 1),
        ({"big.py": 40}, {"big.py": 40}, 0),
        ({"big.py": 41}, {"big.py": 40}, 1),
        ({"nested/deep.py": 40}, {"nested/deep.py": 40}, 0),
    ],
    ids=["at-ceiling", "empty", "one-over", "exempt-within", "exempt-over", "nested-exemption-is-package-relative"],
)
def test_the_ceiling_is_inclusive_and_exemptions_replace_it(
    tmp_path: UPath, modules: dict[str, int], exemptions: dict[str, int], expected_exit_code: int
) -> None:
    package = _write_package(tmp_path, {name: _statements(count) for name, count in modules.items()})
    pyproject = _write_size_pyproject(tmp_path, max_module_statements=10, exemptions=exemptions)

    assert _run_size_gate(package, pyproject) == expected_exit_code


def test_size_report_names_only_the_offending_module(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"small.py": _statements(2), "big.py": _statements(50)})

    _run_size_gate(package, _write_size_pyproject(tmp_path, max_module_statements=10, exemptions={}))

    out = capsys.readouterr().out
    assert "big.py" in out
    assert "small.py" not in out


@pytest.mark.parametrize(
    ("statements", "exemption", "ratchetable"),
    [(25, 40, True), (8, 12, True), (39, 40, False)],
    ids=["shrunk", "under-global-ceiling", "just-under-own-ceiling"],
)
def test_reports_exemptions_that_can_be_lowered(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str], statements: int, exemption: int, ratchetable: bool
) -> None:
    package = _write_package(tmp_path, {"m.py": _statements(statements)})
    pyproject = _write_size_pyproject(tmp_path, max_module_statements=10, exemptions={"m.py": exemption})

    assert _run_size_gate(package, pyproject) == 0
    assert ("can be lowered or deleted" in capsys.readouterr().out) is ratchetable


def test_reports_exemptions_for_modules_that_no_longer_exist(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(2)})
    pyproject = _write_size_pyproject(tmp_path, max_module_statements=10, exemptions={"gone.py": 40})

    assert _run_size_gate(package, pyproject) == 0
    assert "gone.py" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing-package", "missing-pyproject", "unparseable-module"])
def test_exits_with_a_clear_message(tmp_path: UPath, capsys: pytest.CaptureFixture[str], problem: str) -> None:
    package = _write_package(tmp_path, {"broken.py": "def (\n" if problem == "unparseable-module" else "x = 1\n"})
    pyproject = _write_size_pyproject(tmp_path, max_module_statements=10, exemptions={})
    message = {
        "missing-package": "not a directory",
        "missing-pyproject": "not found",
        "unparseable-module": "does not parse",
    }[problem]
    if problem == "missing-package":
        package = tmp_path / "absent"
    if problem == "missing-pyproject":
        pyproject = tmp_path / "absent.toml"

    with pytest.raises(SystemExit) as excinfo:
        _run_size_gate(package, pyproject)

    assert excinfo.value.code == 1
    assert message in capsys.readouterr().err


def test_defaults_the_ceiling_when_the_config_table_is_absent(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(200)})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')

    assert _run_size_gate(package, pyproject) == 1


def test_counts_nested_statements_and_a_long_docstring_as_one() -> None:
    nested = "def f():\n    if True:\n        return 1\n    return 2\n"
    many_lines = '"""Long.\n\n' + "\n".join(f"line {index}" for index in range(30)) + '\n"""\nx = 1\n'

    assert count_statements(nested) == 4
    assert count_statements('"""Short."""\nx = 1\n') == count_statements(many_lines) == 2

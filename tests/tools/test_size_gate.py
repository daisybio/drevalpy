"""Tests for the per-module statement ceiling gate in ``tools/size_gate.py``."""

from __future__ import annotations

import pytest
from upath import UPath

from tools.size_gate import count_statements, main


def _write_package(path: UPath, modules: dict[str, str]) -> UPath:
    """Write a package tree whose modules hold the given source text."""
    package = path / "drevalpy"
    for relative, source in modules.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return package


def _statements(count: int) -> str:
    """Build a module source with exactly ``count`` top-level statements."""
    return "\n".join(f"x{index} = {index}" for index in range(count)) + "\n"


def _write_pyproject(path: UPath, max_module_statements: int, exemptions: dict[str, int]) -> UPath:
    """Write a minimal ``pyproject.toml`` holding a size-gate config table.

    Exemption keys are written package-relative, the form the gate reports.
    """
    lines = ["[tool.drevalpy.size_gate]", f"max_module_statements = {max_module_statements}"]
    if exemptions:
        lines.append("[tool.drevalpy.size_gate.exemptions]")
        lines.extend(f'"drevalpy/{key}" = {value}' for key, value in exemptions.items())
    target = path / "pyproject.toml"
    target.write_text("\n".join(lines))
    return target


def _run(package: UPath, pyproject: UPath) -> int:
    return main(["--package", str(package), "--pyproject", str(pyproject)])


def test_passes_when_every_module_meets_the_ceiling(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(5), "b.py": _statements(10)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    assert _run(package, pyproject) == 0


def test_fails_when_a_module_is_above_the_ceiling(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(5), "b.py": _statements(11)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    assert _run(package, pyproject) == 1


def test_names_the_offending_module_in_the_report(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"small.py": _statements(2), "big.py": _statements(50)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    _run(package, pyproject)

    out = capsys.readouterr().out
    assert "big.py" in out
    assert "small.py" not in out


def test_exempted_module_above_global_ceiling_but_within_its_own_passes(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"big.py": _statements(40)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"big.py": 40})

    assert _run(package, pyproject) == 0


def test_exempted_module_above_its_own_ceiling_fails(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"big.py": _statements(41)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"big.py": 40})

    assert _run(package, pyproject) == 1


def test_reports_a_shrunk_module_as_ratchetable(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"shrunk.py": _statements(25)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"shrunk.py": 40})

    exit_code = _run(package, pyproject)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "can be lowered or deleted" in out
    assert "shrunk.py" in out


def test_reports_a_module_that_fell_under_the_global_ceiling_as_ratchetable(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"split.py": _statements(8)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"split.py": 12})

    exit_code = _run(package, pyproject)

    assert exit_code == 0
    assert "can be lowered or deleted" in capsys.readouterr().out


def test_a_module_just_under_its_own_ceiling_is_not_yet_ratchetable(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"big.py": _statements(39)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"big.py": 40})

    exit_code = _run(package, pyproject)

    assert exit_code == 0
    assert "can be lowered or deleted" not in capsys.readouterr().out


def test_reports_exemptions_for_modules_that_no_longer_exist(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(2)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"gone.py": 40})

    exit_code = _run(package, pyproject)

    assert exit_code == 0
    assert "gone.py" in capsys.readouterr().out


def test_exemption_paths_are_package_relative_not_filesystem_absolute(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"nested/deep.py": _statements(40)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"nested/deep.py": 40})

    assert _run(package, pyproject) == 0


def test_exits_with_a_clear_message_when_the_package_is_missing(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    with pytest.raises(SystemExit) as excinfo:
        _run(tmp_path / "absent", pyproject)

    assert excinfo.value.code == 1
    assert "not a directory" in capsys.readouterr().err


def test_exits_when_the_pyproject_is_missing(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(2)})

    with pytest.raises(SystemExit) as excinfo:
        _run(package, tmp_path / "absent.toml")

    assert excinfo.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_exits_when_a_module_does_not_parse(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"broken.py": "def (\n"})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    with pytest.raises(SystemExit) as excinfo:
        _run(package, pyproject)

    assert excinfo.value.code == 1
    assert "does not parse" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("count", "expected_exit_code"),
    [
        pytest.param(10, 0, id="exactly-at-ceiling-passes"),
        pytest.param(11, 1, id="one-over-ceiling-fails"),
        pytest.param(0, 0, id="empty-module-passes"),
    ],
)
def test_ceiling_is_inclusive(tmp_path: UPath, count: int, expected_exit_code: int) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(count) if count else ""})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})

    assert _run(package, pyproject) == expected_exit_code


def test_defaults_the_ceiling_when_the_config_table_is_absent(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(200)})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')

    assert _run(package, pyproject) == 1


def test_counts_nested_statements_not_just_top_level() -> None:
    source = "def f():\n    if True:\n        return 1\n    return 2\n"

    assert count_statements(source) == 4


def test_a_long_docstring_counts_as_one_statement() -> None:
    one_line = '"""Short."""\nx = 1\n'
    many_lines = '"""Long.\n\n' + "\n".join(f"line {index}" for index in range(30)) + '\n"""\nx = 1\n'

    assert count_statements(one_line) == count_statements(many_lines) == 2

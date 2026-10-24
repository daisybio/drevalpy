"""Tests for the per-module statement ceiling gate in ``tools/size_gate.py``."""

from __future__ import annotations

import pytest
from upath import UPath

from tools.size_gate import count_statements, main


def _write_package(path: UPath, modules: dict[str, str]) -> UPath:
    package = path / "drevalpy"
    for relative, source in modules.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    return package


def _statements(count: int) -> str:
    return "\n".join(f"x{index} = {index}" for index in range(count)) + "\n" if count else ""


def _write_pyproject(path: UPath, max_module_statements: int, exemptions: dict[str, int]) -> UPath:
    """Write a minimal ``pyproject.toml``; exemption keys are package-relative, the form the gate reports."""
    lines = ["[tool.drevalpy.size_gate]", f"max_module_statements = {max_module_statements}"]
    if exemptions:
        lines.append("[tool.drevalpy.size_gate.exemptions]")
        lines.extend(f'"drevalpy/{key}" = {value}' for key, value in exemptions.items())
    target = path / "pyproject.toml"
    target.write_text("\n".join(lines))
    return target


def _run(package: UPath, pyproject: UPath) -> int:
    return main(["--package", str(package), "--pyproject", str(pyproject)])


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
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions=exemptions)

    assert _run(package, pyproject) == expected_exit_code


def test_names_only_the_offending_module_in_the_report(tmp_path: UPath, capsys: pytest.CaptureFixture[str]) -> None:
    package = _write_package(tmp_path, {"small.py": _statements(2), "big.py": _statements(50)})

    _run(package, _write_pyproject(tmp_path, max_module_statements=10, exemptions={}))

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
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"m.py": exemption})

    assert _run(package, pyproject) == 0
    assert ("can be lowered or deleted" in capsys.readouterr().out) is ratchetable


def test_reports_exemptions_for_modules_that_no_longer_exist(
    tmp_path: UPath, capsys: pytest.CaptureFixture[str]
) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(2)})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={"gone.py": 40})

    assert _run(package, pyproject) == 0
    assert "gone.py" in capsys.readouterr().out


@pytest.mark.parametrize("problem", ["missing-package", "missing-pyproject", "unparseable-module"])
def test_exits_with_a_clear_message(tmp_path: UPath, capsys: pytest.CaptureFixture[str], problem: str) -> None:
    package = _write_package(tmp_path, {"broken.py": "def (\n" if problem == "unparseable-module" else "x = 1\n"})
    pyproject = _write_pyproject(tmp_path, max_module_statements=10, exemptions={})
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
        _run(package, pyproject)

    assert excinfo.value.code == 1
    assert message in capsys.readouterr().err


def test_defaults_the_ceiling_when_the_config_table_is_absent(tmp_path: UPath) -> None:
    package = _write_package(tmp_path, {"a.py": _statements(200)})
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n')

    assert _run(package, pyproject) == 1


def test_counts_nested_statements_and_a_long_docstring_as_one() -> None:
    nested = "def f():\n    if True:\n        return 1\n    return 2\n"
    many_lines = '"""Long.\n\n' + "\n".join(f"line {index}" for index in range(30)) + '\n"""\nx = 1\n'

    assert count_statements(nested) == 4
    assert count_statements('"""Short."""\nx = 1\n') == count_statements(many_lines) == 2

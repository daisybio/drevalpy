"""Policy test: every Python module in drevalpy should have a mirrored test file.

This test auto-discovers all ``.py`` files in the package (excluding ``__init__.py``)
and warns when no corresponding ``test_<name>.py`` exists at the mirrored location
under ``tests/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "drevalpy"
TESTS_ROOT = REPO_ROOT / "tests"


def _discover_modules() -> list[str]:
    """Return relative paths (from PACKAGE_ROOT) of all non-init .py files."""
    modules: list[str] = []
    for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        modules.append(str(py_file.relative_to(PACKAGE_ROOT)))
    return modules


def _mirrored_test_path(relative_module: str) -> Path:
    rel = Path(relative_module)
    return TESTS_ROOT / rel.parent / f"test_{rel.name}"


ALL_MODULES = _discover_modules()


@pytest.mark.parametrize("relative_module", ALL_MODULES)
def test_module_has_mirrored_test(relative_module: str) -> None:
    """Warn (not fail) when a source module lacks a mirrored test file."""
    module_path = PACKAGE_ROOT / relative_module
    assert module_path.is_file(), f"missing source module: {relative_module}"
    expected = _mirrored_test_path(relative_module)
    if not expected.is_file():
        warnings.warn(
            f"Missing test mirror for {relative_module}: expected {expected.relative_to(REPO_ROOT)}",
            stacklevel=1,
        )

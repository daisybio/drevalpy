"""Policy test: every module in ``drevalpy`` has a mirrored test file (see "Test layout" in ``AGENTS.md``)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "drevalpy"
TESTS_ROOT = REPO_ROOT / "tests"

#: ``_make_gene_lists.py`` is a maintenance script, not library code: nothing imports it and
#: coverage omits it.
EXEMPT_MODULES: frozenset[str] = frozenset(
    {
        "components/featurizers/sample/gene_lists/_make_gene_lists.py",
    }
)


def _discover_modules() -> list[str]:
    modules = []
    for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        relative = py_file.relative_to(PACKAGE_ROOT).as_posix()
        if relative in EXEMPT_MODULES:
            continue
        modules.append(relative)
    return modules


def _candidate_test_paths(relative_module: str) -> list[Path]:
    rel = Path(relative_module)
    mirrored_dir = TESTS_ROOT / rel.parent
    candidates = [mirrored_dir / f"test_{rel.name}"]
    stripped = rel.name.lstrip("_")
    if stripped != rel.name:
        candidates.append(mirrored_dir / f"test_{stripped}")
    return candidates


ALL_MODULES = _discover_modules()


def test_exempt_modules_still_exist() -> None:
    stale = sorted(name for name in EXEMPT_MODULES if not (PACKAGE_ROOT / name).is_file())
    assert not stale, f"EXEMPT_MODULES lists modules that no longer exist: {stale}"


@pytest.mark.parametrize("relative_module", ALL_MODULES)
def test_module_has_mirrored_test(relative_module: str) -> None:
    assert (PACKAGE_ROOT / relative_module).is_file(), f"missing source module: {relative_module}"

    candidates = _candidate_test_paths(relative_module)
    if any(candidate.is_file() for candidate in candidates):
        return

    expected = " or ".join(str(candidate.relative_to(REPO_ROOT)) for candidate in candidates)
    pytest.fail(
        f"Missing test mirror for drevalpy/{relative_module}: expected {expected}. "
        "Add a real test file at that path; do not add a stub to satisfy this guard."
    )

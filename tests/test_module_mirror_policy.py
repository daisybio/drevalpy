"""Policy test: every module in ``drevalpy`` has a mirrored test file.

The mirroring convention is documented in ``AGENTS.md``. This guard enforces it
so a new source module cannot land without a test file at the mirrored location.

Naming, following ``AGENTS.md`` rules 1-4:

* A public module ``drevalpy/a/b/c.py`` requires ``tests/a/b/test_c.py``.
* A private module ``_c.py`` is satisfied by *either* ``test__c.py`` or the
  underscore-stripped ``test_c.py``. The stripped form is the house style - see
  ``tests/models/config/`` and ``tests/registry/`` - but both are accepted so a
  literal mirror is never wrong.
* ``__init__.py`` is not checked here. Package surfaces are tested in
  ``test_init.py`` where they re-export something worth pinning, but plenty of
  ``__init__.py`` files hold nothing but imports, and demanding a file for each
  would push the suite towards exactly the stub mirrors ``AGENTS.md`` forbids.

Only the package tree is walked, so the repository's other Python - ``tools/``,
``docs/`` generators, ``tests/`` itself - is out of scope by construction. The
``EXEMPT_MODULES`` list below is for modules inside the package that are
deliberately not mirrored; keep it as short as it is now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "drevalpy"
TESTS_ROOT = REPO_ROOT / "tests"

#: Modules inside the package that intentionally have no mirrored test.
#:
#: ``_make_gene_lists.py`` is a maintenance script used to regenerate the
#: packaged gene-list CSVs from an upstream annotation dump. It is not imported
#: by any shipped code path and is excluded from coverage measurement via
#: ``[tool.coverage.run].omit`` in ``pyproject.toml``; mirroring it would test
#: the tooling rather than the library.
EXEMPT_MODULES: frozenset[str] = frozenset(
    {
        "components/featurizers/cell_line/gene_lists/_make_gene_lists.py",
    }
)


def _discover_modules() -> list[str]:
    """Return every non-``__init__`` module path, relative to the package root.

    Returns:
        Sorted POSIX-style relative paths, excluding :data:`EXEMPT_MODULES`.
    """
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
    """Return the acceptable mirrored test paths for one module.

    Args:
        relative_module: Module path relative to the package root.

    Returns:
        ``[test_<name>.py]`` for a public module; for a private module also the
        underscore-stripped spelling, in the order they should be reported.
    """
    rel = Path(relative_module)
    mirrored_dir = TESTS_ROOT / rel.parent
    candidates = [mirrored_dir / f"test_{rel.name}"]
    stripped = rel.name.lstrip("_")
    if stripped != rel.name:
        candidates.append(mirrored_dir / f"test_{stripped}")
    return candidates


ALL_MODULES = _discover_modules()


def test_exempt_modules_still_exist() -> None:
    """Keep :data:`EXEMPT_MODULES` from silently outliving its modules."""
    stale = sorted(name for name in EXEMPT_MODULES if not (PACKAGE_ROOT / name).is_file())
    assert not stale, f"EXEMPT_MODULES lists modules that no longer exist: {stale}"


@pytest.mark.parametrize("relative_module", ALL_MODULES)
def test_module_has_mirrored_test(relative_module: str) -> None:
    """Fail when a source module has no mirrored test file."""
    assert (PACKAGE_ROOT / relative_module).is_file(), f"missing source module: {relative_module}"

    candidates = _candidate_test_paths(relative_module)
    if any(candidate.is_file() for candidate in candidates):
        return

    expected = " or ".join(str(candidate.relative_to(REPO_ROOT)) for candidate in candidates)
    pytest.fail(
        f"Missing test mirror for drevalpy/{relative_module}: expected {expected}. "
        "See the test-layout section of AGENTS.md; do not add a stub to satisfy this guard."
    )

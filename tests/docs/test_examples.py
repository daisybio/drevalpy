"""Guard the runnable examples the documentation includes.

The how-to pages show ``docs/examples/*.py`` with ``literalinclude``, which
keeps page and code identical but proves nothing about either: a
``literalinclude`` of a module that no longer imports renders happily. The docs
build closes that gap by calling ``docs/_examples.verify_documented_examples``,
and this module runs the same verification under pytest so a broken example does
not have to wait for a docs build to be noticed.

Importing the examples registers components, which would break the exact-count
assertions in ``tests/test_featurizer_block_policy.py``. The verification
therefore runs in a subprocess; only the RST cross-checks happen in-process,
where they touch no registry.

Both facts the subprocess establishes - what the examples register, and that the
registries look the same afterwards - start from the same pristine interpreter,
so one child process reports both and the two tests assert on their own half of
its output.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._trusted_subprocess import run_trusted_python
from tests.docs.test_docs_structure import built_pages, exclude_patterns

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
EXAMPLES = DOCS / "examples"
EXTEND_PAGES = DOCS / "how-to" / "extend"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

#: TEMPORARY(D3): the literal includes still sit on this page of the old tree, which
#: ``docs/conf.py`` excludes from the build. D3 moves them to ``how-to/extend/``; set
#: this to ``()`` then. With it empty, only built pages count as including an example.
#: A path that no longer exists (the old tree is deleted) is ignored.
TEMPORARY_INCLUDERS: tuple[Path, ...] = (DOCS / "python" / "extensions.rst",)

_LITERALINCLUDE = re.compile(r"^\s*\.\. literalinclude:: (\S+)\s*$", re.MULTILINE)

#: Runs the docs build's own verification in a fresh interpreter and reports the
#: registry sizes on either side of it along with what the examples registered.
#: The registry imports come before ``_examples`` so ``before`` is a genuine
#: pre-verification reading.
_VERIFY_SCRIPT = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(DOCS)!r})\n"
    "from drevalpy.registry import sample_featurizer, drug_featurizer, predictor\n"
    "before = [len(sample_featurizer.list()), len(drug_featurizer.list()), len(predictor.list())]\n"
    "from _examples import verify_documented_examples\n"
    "registered = {k: list(v) for k, v in verify_documented_examples().items()}\n"
    "after = [len(sample_featurizer.list()), len(drug_featurizer.list()), len(predictor.list())]\n"
    "sys.stdout.write(json.dumps({'registered': registered, 'before': before, 'after': after}))\n"
)


@pytest.fixture(scope="module")
def verification() -> dict[str, Any]:
    """Report of one pristine-interpreter run of ``verify_documented_examples``.

    :returns: ``{"registered": ..., "before": ..., "after": ...}`` as reported by
        the child process.
    """
    completed = run_trusted_python(_VERIFY_SCRIPT, cwd=str(REPO_ROOT))
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _includer_pages() -> list[Path]:
    """Return the pages whose ``literalinclude`` lines count: built pages plus the temporary allowance."""
    return built_pages(DOCS, exclude_patterns(DOCS)) + [path for path in TEMPORARY_INCLUDERS if path.is_file()]


def _literalincludes(pages: list[Path]) -> list[tuple[Path, Path]]:
    """Return ``(page, target)`` for every ``literalinclude``, indented (inside a tab) or not.

    Targets are resolved the way Sphinx does: ``/x`` from the docs root, anything else from the page.
    """
    found = []
    for page in pages:
        for target in _LITERALINCLUDE.findall(page.read_text(encoding="utf-8")):
            base = DOCS if target.startswith("/") else page.parent
            found.append((page, Path(os.path.normpath(base / target.lstrip("/")))))
    return found


class TestVerificationInAPristineInterpreter:
    """Extended tier: the shared ``verification`` fixture spawns an interpreter.

    Both tests read the same child-process report, so the ~2.4s is only saved when
    both are deselected - hence a class-level marker rather than two per-test ones.
    """

    pytestmark = pytest.mark.slow

    def test_examples_import_register_and_conform(self, verification: dict[str, Any]) -> None:
        """The examples import, register what is expected, and pass every check."""
        from _examples import EXPECTED_REGISTRATIONS

        observed = {key: tuple(value) for key, value in verification["registered"].items()}
        assert observed == EXPECTED_REGISTRATIONS

    def test_verifying_the_examples_leaves_the_registries_alone(self, verification: dict[str, Any]) -> None:
        """The verification must not be observable afterwards.

        ``tests/test_featurizer_block_policy.py`` asserts exact registry counts, and
        the docs build's generated component catalogs do the same, so an example that
        stayed registered would break both.
        """
        assert verification["after"] == verification["before"]


def test_every_example_module_is_shown_on_some_page() -> None:
    """An example nobody reads is dead code; some page under ``docs/`` must include each one."""
    from _examples import EXAMPLE_MODULES

    included = {target for _, target in _literalincludes(_includer_pages())}
    # toy_conformance.py is the harness that checks the others, so it is
    # described in prose rather than shown; everything else is on a page.
    expected = {EXAMPLES / f"{name}.py" for name in EXAMPLE_MODULES} - {EXAMPLES / "toy_conformance.py"}
    assert expected <= included, f"Example modules no page includes: {sorted(p.name for p in expected - included)}"


def test_pages_only_include_files_that_exist() -> None:
    """A renamed example must not leave a dangling ``literalinclude``."""
    missing = [
        f"{page.relative_to(DOCS).as_posix()}: {target}"
        for page, target in _literalincludes(_includer_pages())
        if not target.is_file()
    ]
    assert not missing, f"literalincludes of missing files: {missing}"


def test_every_example_module_is_listed_in_the_driver() -> None:
    """A module the driver does not import is never checked."""
    from _examples import EXAMPLE_MODULES

    on_disk = {path.stem for path in EXAMPLES.glob("*.py") if path.name != "__init__.py"}
    assert on_disk == set(EXAMPLE_MODULES)


#: Import paths and hooks the extension pages used to teach that never existed or no longer
#: do. Kept as a regression guard because every one of them was published.
DEAD_REFERENCES = (
    "drevalpy.components.core",
    "drevalpy.components.featurizers.sample.base",
    "drevalpy.components.predictors.abstract",
    "drevalpy.registry.sample_featurizer import register",
    "drevalpy.registry.predictor import register",
    "drevalpy.visualization.base import Visualization",
)


def _extend_pages() -> list[Path]:
    return sorted(EXTEND_PAGES.rglob("*.rst"))


def test_extend_pages_do_not_teach_dead_import_paths() -> None:
    pages = _extend_pages() + [path for path in TEMPORARY_INCLUDERS if path.is_file()]
    found = [
        f"{page.relative_to(DOCS).as_posix()}: {reference}"
        for page in pages
        for reference in DEAD_REFERENCES
        if reference in page.read_text(encoding="utf-8")
    ]
    assert not found, f"pages point at unsupported import paths: {found}"


def test_extend_pages_with_python_route_plugin_authors_through_the_supported_surface() -> None:
    """Once the extend pages show Python, they must teach ``drevalpy.plugin`` and ``drevalpy.testing``.

    Vacuous while the pages are stubs, so the check cannot fail the Wave 1 tree.
    """
    from _snippets import collect

    pages = _extend_pages()
    if not any(snippet.language == "python" for page in pages for snippet in collect(page)):
        pytest.skip("the extend pages show no Python yet")
    text = "\n".join(page.read_text(encoding="utf-8") for page in pages)
    assert "drevalpy.plugin" in text
    assert "drevalpy.testing" in text

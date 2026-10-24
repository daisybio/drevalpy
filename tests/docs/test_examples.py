"""Guard the runnable examples the documentation includes.

``literalinclude`` keeps page and code identical but proves nothing about either, so this runs
``docs/_examples.verify_documented_examples`` under pytest. Importing the examples registers components,
so that verification runs in a subprocess; only the RST cross-checks happen in-process.
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

from _snippets import collect  # noqa: E402

_LITERALINCLUDE = re.compile(r"^\s*\.\. literalinclude:: (\S+)\s*$", re.MULTILINE)

#: Runs the docs build's verification in a fresh interpreter, reporting registry sizes before and after.
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
    """One pristine-interpreter run of ``verify_documented_examples``: registered, before, after."""
    completed = run_trusted_python(_VERIFY_SCRIPT, cwd=str(REPO_ROOT))
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _includer_pages() -> list[Path]:
    """The built pages."""
    return built_pages(DOCS, exclude_patterns(DOCS))


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


@pytest.mark.slow
class TestVerificationInAPristineInterpreter:
    def test_examples_import_register_and_conform(self, verification: dict[str, Any]) -> None:
        from _examples import EXPECTED_REGISTRATIONS

        observed = {key: tuple(value) for key, value in verification["registered"].items()}
        assert observed == EXPECTED_REGISTRATIONS

    def test_verifying_the_examples_leaves_the_registries_alone(self, verification: dict[str, Any]) -> None:
        assert verification["after"] == verification["before"]


def test_every_example_module_is_shown_on_some_page() -> None:
    from _examples import EXAMPLE_MODULES

    included = {target for _, target in _literalincludes(_includer_pages())}
    # toy_conformance.py is the harness, described in prose rather than shown.
    expected = {EXAMPLES / f"{name}.py" for name in EXAMPLE_MODULES} - {EXAMPLES / "toy_conformance.py"}
    assert expected <= included, f"Example modules no page includes: {sorted(p.name for p in expected - included)}"


def test_pages_only_include_files_that_exist() -> None:
    missing = [
        f"{page.relative_to(DOCS).as_posix()}: {target}"
        for page, target in _literalincludes(_includer_pages())
        if not target.is_file()
    ]
    assert not missing, f"literalincludes of missing files: {missing}"


def test_every_example_module_is_listed_in_the_driver() -> None:
    from _examples import EXAMPLE_MODULES

    on_disk = {path.stem for path in EXAMPLES.glob("*.py") if path.name != "__init__.py"}
    assert on_disk == set(EXAMPLE_MODULES)


#: Import paths the extension pages once taught that never existed or no longer do.
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
    pages = _extend_pages()
    found = [
        f"{page.relative_to(DOCS).as_posix()}: {reference}"
        for page in pages
        for reference in DEAD_REFERENCES
        if reference in page.read_text(encoding="utf-8")
    ]
    assert not found, f"pages point at unsupported import paths: {found}"


def test_extend_pages_with_python_route_plugin_authors_through_the_supported_surface() -> None:
    pages = _extend_pages()
    assert any(snippet.language == "python" for page in pages for snippet in collect(page))
    text = "\n".join(page.read_text(encoding="utf-8") for page in pages)
    assert "drevalpy.plugin" in text
    assert "drevalpy.testing" in text

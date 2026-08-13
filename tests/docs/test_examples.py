"""Guard the runnable examples the extensions page includes.

``docs/python/extensions.rst`` shows its examples with ``literalinclude``, which
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
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
EXAMPLES = DOCS / "examples"
EXTENSIONS_PAGE = DOCS / "python" / "extensions.rst"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

#: Runs the docs build's own verification in a fresh interpreter and reports the
#: registry sizes on either side of it along with what the examples registered.
#: The registry imports come before ``_examples`` so ``before`` is a genuine
#: pre-verification reading.
_VERIFY_SCRIPT = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(DOCS)!r})\n"
    "from drevalpy.registry import cell_line_featurizer, drug_featurizer, predictor\n"
    "before = [len(cell_line_featurizer.list()), len(drug_featurizer.list()), len(predictor.list())]\n"
    "from _examples import verify_documented_examples\n"
    "registered = {k: list(v) for k, v in verify_documented_examples().items()}\n"
    "after = [len(cell_line_featurizer.list()), len(drug_featurizer.list()), len(predictor.list())]\n"
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


def _literalincluded_paths() -> list[str]:
    """Return every path the page pulls in, indented (inside a tab) or not."""
    text = EXTENSIONS_PAGE.read_text(encoding="utf-8")
    return re.findall(r"^\s*\.\. literalinclude:: /(examples/\S+)$", text, flags=re.MULTILINE)


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


def test_every_example_module_is_shown_on_the_page() -> None:
    """An example nobody reads is dead code; the page must include each one."""
    from _examples import EXAMPLE_MODULES

    included = set(_literalincluded_paths())
    # toy_conformance.py is the harness that checks the others, so it is
    # described in prose rather than shown; everything else is on the page.
    expected = {f"examples/{name}.py" for name in EXAMPLE_MODULES} - {"examples/toy_conformance.py"}
    assert expected <= included, f"Example modules missing from extensions.rst: {sorted(expected - included)}"


def test_the_page_only_includes_files_that_exist() -> None:
    """A renamed example must not leave a dangling ``literalinclude``."""
    missing = [target for target in _literalincluded_paths() if not (DOCS / target).is_file()]
    assert not missing, f"extensions.rst literalincludes missing files: {missing}"


def test_every_example_module_is_listed_in_the_driver() -> None:
    """A module the driver does not import is never checked."""
    from _examples import EXAMPLE_MODULES

    on_disk = {path.stem for path in EXAMPLES.glob("*.py") if path.name != "__init__.py"}
    assert on_disk == set(EXAMPLE_MODULES)


#: Import paths and hooks the page used to teach that never existed or no longer
#: do. Kept as a regression guard because every one of them was published.
DEAD_REFERENCES = (
    "drevalpy.components.core",
    "drevalpy.components.featurizers.cell_line.base",
    "drevalpy.components.predictors.abstract",
    "drevalpy.registry.cell_line_featurizer import register",
    "drevalpy.registry.predictor import register",
    "drevalpy.visualization.base import Visualization",
)


def test_the_page_teaches_the_supported_import_surface() -> None:
    """Prose and snippets must route plugin authors through ``drevalpy.plugin``."""
    text = EXTENSIONS_PAGE.read_text(encoding="utf-8")
    found = [reference for reference in DEAD_REFERENCES if reference in text]
    assert not found, f"extensions.rst points at unsupported import paths: {found}"
    assert "drevalpy.plugin" in text
    assert "drevalpy.testing" in text

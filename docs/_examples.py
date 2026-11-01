"""Import the runnable plugin examples so the docs build fails when one rots.

``docs/examples/`` holds real plugin components. The extensions page shows them
with ``literalinclude``, which keeps the page and the code identical but proves
nothing about either -- a ``literalinclude`` of a file that no longer imports
renders happily. So the build imports every example, runs drevalpy's shipped
conformance checks over them, and generates the registry table the page shows
from the registries the examples actually landed in.

Registering mutates process-wide state, so the registries are rolled back once
the check has passed: the examples exist to be read, not to appear in the
generated component catalogs.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _generated_io import write_text_if_changed

from drevalpy.registry._all import all_registries, restore_all, snapshot_all

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
GENERATED_EXAMPLES = DOCS_DIR / "reference" / "_generated_examples.rst"

#: Import package of the examples. ``docs`` has no ``__init__.py`` and resolves
#: as a namespace package, which keeps the generic name ``examples`` off the top
#: level of every process that builds the docs.
EXAMPLE_PACKAGE = "docs.examples"

#: Every example module, in the order the extensions page presents them.
EXAMPLE_MODULES: tuple[str, ...] = (
    "toy_sample_featurizer",
    "toy_drug_featurizer",
    "toy_mean_predictor",
    "toy_ridge_predictor",
    "toy_block_predictor",
    "toy_splitter",
    "toy_visualization",
    "toy_conformance",
)

#: What importing the examples must add to each registry. Pinned rather than
#: derived so a decorator that silently stops running is a build failure.
EXPECTED_REGISTRATIONS: dict[str, tuple[str, ...]] = {
    "sample_featurizer": ("toySample",),
    "drug_featurizer": ("toyDrugHash",),
    "predictor": ("toyBlockRidge", "toyMean", "toyRidge"),
    "splitter": ("TOY_LSO",),
    "visualization": ("toyResiduals",),
}

_LABELS = {
    "sample_featurizer": "Sample featurizer",
    "drug_featurizer": "Drug featurizer",
    "predictor": "Predictor",
    "splitter": "Split mode",
    "visualization": "Visualization",
}


def _names() -> dict[str, frozenset[str]]:
    registries = all_registries()
    return {kind: frozenset(registries[kind].names()) for kind in EXPECTED_REGISTRATIONS}


def _import_examples() -> None:
    """Import every example, putting the repository root on the path first.

    ``docs.examples`` resolves relative to the repository root, which the
    editable install deliberately no longer places on ``sys.path``. Adding it
    here keeps the driver independent of the directory Sphinx was invoked from.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    for name in EXAMPLE_MODULES:
        importlib.import_module(f"{EXAMPLE_PACKAGE}.{name}")


def _registered_by_examples(before: dict[str, frozenset[str]]) -> dict[str, tuple[str, ...]]:
    after = _names()
    return {kind: tuple(sorted(after[kind] - before[kind])) for kind in EXPECTED_REGISTRATIONS}


def _assert_expected(observed: dict[str, tuple[str, ...]]) -> None:
    if observed != EXPECTED_REGISTRATIONS:
        msg = (
            "The documented examples no longer register what docs/_examples.py expects: "
            f"expected {EXPECTED_REGISTRATIONS}, got {observed}. Update EXPECTED_REGISTRATIONS "
            "and the extensions page together."
        )
        raise RuntimeError(msg)


def _run_checks() -> None:
    conformance = importlib.import_module(f"{EXAMPLE_PACKAGE}.toy_conformance")
    conformance.check_components()
    conformance.check_splitter()
    conformance.check_visualization()


def _describe(registry_name: str, name: str) -> str:
    metadata: Mapping[str, Any] = all_registries()[registry_name].metadata(name)
    return " ".join((metadata.get("description") or "").split()) or "No description."


def _render(observed: dict[str, tuple[str, ...]]) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 26 50",
        "",
        "   * - Registered name",
        "     - Registry",
        "     - Description",
    ]
    for registry_name in EXPECTED_REGISTRATIONS:
        for name in observed[registry_name]:
            lines.extend(
                [
                    f"   * - ``{name}``",
                    f"     - {_LABELS[registry_name]}",
                    f"     - {_describe(registry_name, name)}",
                ]
            )
    return "\n".join([*lines, ""])


def verify_documented_examples() -> dict[str, tuple[str, ...]]:
    """Import, check and catalog the examples, then undo their registrations.

    :returns: Registry name mapped to the names the examples registered.
    :raises RuntimeError: If an example fails to import, registers something
        other than expected, or fails a conformance check.
    """
    snapshot = snapshot_all()
    before = _names()
    try:
        _import_examples()
        observed = _registered_by_examples(before)
        _assert_expected(observed)
        _run_checks()
        write_text_if_changed(GENERATED_EXAMPLES, _render(observed))
    except Exception as exc:
        msg = (
            "A documented plugin example under docs/examples/ no longer works, so the "
            "extensions page would teach code that does not run. Fix the example (or the "
            "library change that broke it) rather than removing this check."
        )
        raise RuntimeError(msg) from exc
    finally:
        restore_all(snapshot)
    return observed

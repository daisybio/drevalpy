"""Layering policy: ``drevalpy.data`` must never depend on ``drevalpy.components``."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "drevalpy"
DATASETS_ROOT = PACKAGE_ROOT / "datasets"

# Absolute references to the components layer, e.g. ``from drevalpy.components.x import y``,
# ``import drevalpy.components.x`` or ``from drevalpy import components``.
_ABSOLUTE_COMPONENTS = re.compile(r"drevalpy\.components\b|from\s+drevalpy\s+import\s+[^\n]*\bcomponents\b")

# Relative references that escape into the sibling components package, e.g. ``from ..components import x``.
# Deliberately anchored on the ``from``/``import`` keyword so that unrelated dotted names such as
# ``networkx.algorithms.components.connected`` do not match.
_RELATIVE_COMPONENTS = re.compile(r"^\s*from\s+\.+components\b", re.MULTILINE)


def test_datasets_layer_does_not_import_components() -> None:
    """``datasets`` does the loading, ``components`` decides what to load - never the reverse."""
    hits = []
    for path in sorted(DATASETS_ROOT.rglob("*.py")):
        content = path.read_text(encoding="utf-8")
        relative = path.relative_to(PACKAGE_ROOT.parent)
        if _ABSOLUTE_COMPONENTS.search(content):
            hits.append(f"{relative}: absolute import of drevalpy.components")
        if _RELATIVE_COMPONENTS.search(content):
            hits.append(f"{relative}: relative import of the components package")
    assert not hits, "drevalpy/datasets must not depend on drevalpy/components: " + "; ".join(hits)


def test_layering_check_detects_violations() -> None:
    """Guard the guard: the patterns must actually fire on the forms we care about."""
    assert _ABSOLUTE_COMPONENTS.search("from drevalpy.components.registry import get_drug_featurizer")
    assert _ABSOLUTE_COMPONENTS.search("import drevalpy.components")
    assert _ABSOLUTE_COMPONENTS.search("from drevalpy import components")
    assert _RELATIVE_COMPONENTS.search("from ..components import featurizers")
    assert _RELATIVE_COMPONENTS.search("from .components.base import Featurizer")
    # Unrelated third-party dotted paths must not be flagged.
    assert not _ABSOLUTE_COMPONENTS.search("import networkx.algorithms.components.connected as nxacc")
    assert not _RELATIVE_COMPONENTS.search("import networkx.algorithms.components.connected as nxacc")

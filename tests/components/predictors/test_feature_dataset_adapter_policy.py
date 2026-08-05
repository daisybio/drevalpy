"""Policy: FeatureDataset-protocol adapter bases are literature-only in-tree."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "drevalpy"
PREDICTORS_ROOT = PACKAGE_ROOT / "components" / "predictors"
LITERATURE_ROOT = PREDICTORS_ROOT / "literature"

ADAPTER_BASES = frozenset({"FeatureDatasetBlockPredictor", "SingleDrugBlockPredictor"})
ALLOWED_BASE_MODULES = frozenset(
    {
        "feature_dataset_block.py",
        "single_drug_block.py",
    }
)


def _class_bases(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def test_adapter_subclasses_live_under_literature() -> None:
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name in ALLOWED_BASE_MODULES and path.parent == PREDICTORS_ROOT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not (_class_bases(node) & ADAPTER_BASES):
                continue
            try:
                path.relative_to(LITERATURE_ROOT)
            except ValueError:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "adapter subclasses must live under predictors/literature:\n" + "\n".join(offenders)

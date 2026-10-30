"""Policy test: the simplified architecture stays simple.

Each check guards against one pattern the v2 simplification removed: deep featurizer hierarchies,
attribute lookups built from ``required_*`` names, hand-written component state methods, parallel
registry implementations and module docstrings describing the removed ``_discover_modules`` scan.
The checks read the source tree (or the populated featurizer registry) and need no data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import drevalpy
from drevalpy.registry.featurizer import featurizers

SRC = Path(drevalpy.__file__).parent
COMPONENTS = SRC / "components"

#: Upper bound on a featurizer's drevalpy-defined MRO, the class itself included.
MAX_FEATURIZER_DEPTH = 4

#: Registry-like classes that are allowed besides the generic ``Registry``; the dataset registry
#: persists its entries to the user config file and only mirrors ``Registry``'s method names.
ALLOWED_REGISTRY_CLASSES = {"Registry", "DatasetRegistry"}


def _modules(root: Path) -> list[tuple[Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))) for path in sorted(root.rglob("*.py"))
    ]


SRC_MODULES = _modules(SRC)


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(SRC.parent)}:{getattr(node, 'lineno', 0)}"


def _drevalpy_mro(cls: type) -> list[str]:
    return [klass.__qualname__ for klass in cls.__mro__ if klass.__module__.split(".")[0] == "drevalpy"]


@pytest.mark.parametrize("name", featurizers.names())
def test_featurizer_hierarchies_stay_shallow(name: str) -> None:
    mro = _drevalpy_mro(featurizers.get(name))

    assert len(mro) <= MAX_FEATURIZER_DEPTH, f"{name}: {' -> '.join(mro)}"


def test_no_attribute_lookup_is_built_from_a_required_name() -> None:
    offenders = []
    for path, tree in SRC_MODULES:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.JoinedStr):
                continue
            literal = "".join(part.value for part in node.args[1].values if isinstance(part, ast.Constant))
            if "required_" in literal:
                offenders.append(_where(path, node))

    assert offenders == []


def test_components_define_no_state_methods() -> None:
    offenders = [
        f"{_where(path, node)} {node.name}"
        for path, tree in _modules(COMPONENTS)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in {"get_state", "set_state"}
    ]

    assert offenders == []


def test_there_is_one_generic_registry_class() -> None:
    registry_classes = [
        (node.name, _where(path, node))
        for path, tree in SRC_MODULES
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.endswith("Registry")
    ]

    assert sum(name == "Registry" for name, _ in registry_classes) == 1, registry_classes
    assert {name for name, _ in registry_classes} <= ALLOWED_REGISTRY_CLASSES, registry_classes


def test_no_module_docstring_mentions_the_removed_module_scan() -> None:
    offenders = [
        str(path.relative_to(SRC.parent))
        for path, tree in SRC_MODULES
        if "_discover_modules" in (ast.get_docstring(tree) or "")
    ]

    assert offenders == []

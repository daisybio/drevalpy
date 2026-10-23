"""Layering policy: ``drevalpy.data`` must not depend on ``drevalpy.components`` at import time."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "drevalpy"
DATA_ROOT = PACKAGE_ROOT / "data"
FORBIDDEN = "drevalpy.components"


def _module_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Collect the imports that run when a module is imported.

    Walks the module body and the bodies of module-level ``if``/``try``/``with`` blocks, and never
    descends into function or class bodies, so a function-local import is not reported.

    :param tree: Parsed module.
    :returns: Every ``import`` and ``from ... import`` statement executed at module scope.
    """
    found: list[ast.Import | ast.ImportFrom] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.append(node)
        elif isinstance(node, ast.If | ast.Try | ast.With):
            for field in ("body", "orelse", "finalbody"):
                pending.extend(getattr(node, field, []))
            for handler in getattr(node, "handlers", []):
                pending.extend(handler.body)
    return found


def _imported_modules(node: ast.Import | ast.ImportFrom, package: str) -> list[str]:
    """Resolve the absolute dotted names an import statement refers to.

    :param node: Import statement.
    :param package: Dotted name of the package that contains the importing module, used to resolve
        relative imports.
    :returns: Absolute module names; ``from pkg import name`` also yields ``pkg.name``.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        parts = package.split(".")
        base = ".".join(parts[: len(parts) - node.level + 1] + ([node.module] if node.module else []))
    else:
        base = node.module or ""
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def find_layer_violations(source: str, package: str, forbidden: str = FORBIDDEN) -> list[str]:
    """List module-level imports of ``forbidden`` in a module's source.

    :param source: Python source of the module.
    :param package: Dotted name of the package containing the module.
    :param forbidden: Dotted name of the package that must not be imported.
    :returns: One ``line N: <name>`` entry per offending import.
    """
    hits = []
    for node in _module_level_imports(ast.parse(source)):
        for name in _imported_modules(node, package):
            if name == forbidden or name.startswith(forbidden + "."):
                hits.append(f"line {node.lineno}: {name}")
                break
    return hits


def test_data_layer_does_not_import_components_at_module_level() -> None:
    """``data`` loads and builds datasets, ``components`` decides what to load - never the reverse."""
    paths = sorted(DATA_ROOT.rglob("*.py"))
    assert paths, f"no modules found under {DATA_ROOT}"
    hits = []
    for path in paths:
        package = ".".join(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts[:-1])
        hits.extend(
            f"{path.relative_to(PACKAGE_ROOT.parent)}: {hit}"
            for hit in find_layer_violations(path.read_text(encoding="utf-8"), package)
        )
    assert not hits, "drevalpy/data must not import drevalpy/components at module level: " + "; ".join(hits)


def test_layering_check_detects_violations() -> None:
    """Guard the guard: every module-level import form must fire."""
    package = "drevalpy.data"
    assert find_layer_violations("from drevalpy.components.registry import get_featurizer", package)
    assert find_layer_violations("import drevalpy.components.featurizers", package)
    assert find_layer_violations("from drevalpy import components", package)
    assert find_layer_violations("from ..components import featurizers", package)
    assert find_layer_violations("from ..components.base import Featurizer", package)
    assert find_layer_violations("try:\n    from drevalpy.components import x\nexcept ImportError:\n    pass", package)


def test_layering_check_ignores_function_local_and_unrelated_imports() -> None:
    """Function-local imports are allowed, and similar dotted names must not be flagged."""
    package = "drevalpy.data"
    assert not find_layer_violations("def f():\n    from drevalpy.components.x import y\n", package)
    assert not find_layer_violations("class C:\n    def f(self):\n        import drevalpy.components\n", package)
    assert not find_layer_violations("import networkx.algorithms.components.connected as nxacc", package)
    assert not find_layer_violations("from .utils import components", package)
    assert not find_layer_violations("from drevalpy.componentsx import y", package)

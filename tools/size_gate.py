"""Enforce a per-module statement ceiling, so split-up modules stay split up.

The refactoring that produced ``models/mixins/`` and the featurizer mixins moved
behaviour out of files that had grown to carry four unrelated concerns each. This
gate is what keeps them from growing back: it counts AST statements per module and
fails when one exceeds its effective ceiling.

Statements rather than lines, because the count is then independent of formatting
and of the Google-style docstrings this codebase requires - a docstring is one
``Expr`` statement whether it is one line or thirty.

The ceiling is ``[tool.drevalpy.size_gate].max_module_statements`` for every
module, except those listed in the ``exemptions`` table, which carry their own
higher ceiling. Every exemption is technical debt: split the module and delete the
entry. A recorded ceiling may be lowered as a module shrinks, never raised to let
a regression through - the same rule ``tools/coverage_gate.py`` follows.

This is deliberately *not* a code-health score. Repowise's ``hotspot_health``
would be the richer signal, but two thirds of its impact is derived from commit
history, which makes it drift with repository activity rather than with the change
under review, and it reads a full 10.0 on the shallow checkout CI does by default.
What this gate measures instead is computable from the working tree alone, so it
means the same thing locally and in CI.

Usage::

    uv run python tools/size_gate.py
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from typing import NamedTuple, NoReturn

from upath import UPath

DEFAULT_PACKAGE = "drevalpy"
DEFAULT_PYPROJECT = "pyproject.toml"
DEFAULT_MAX_MODULE_STATEMENTS = 150
RATCHET_MARGIN = 10


class GateConfig(NamedTuple):
    """Resolved ``[tool.drevalpy.size_gate]`` settings."""

    max_module_statements: int
    exemptions: dict[str, int]


class ModuleSize(NamedTuple):
    """One module's measured statement count and the ceiling it is held to."""

    path: str
    statements: int
    ceiling: int
    exempt: bool


def load_gate_config(pyproject: UPath) -> GateConfig:
    """Read the size-gate configuration from ``pyproject.toml``.

    Args:
        pyproject: Path to the ``pyproject.toml`` holding the config table.

    Returns:
        The resolved configuration, with defaults applied when the table or any
        of its keys are absent.

    Raises:
        SystemExit: If ``pyproject`` does not exist.
    """
    if not pyproject.is_file():
        _fail(f"{pyproject} not found; run the gate from the repository root.")
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    table = data.get("tool", {}).get("drevalpy", {}).get("size_gate", {})
    exemptions = {_normalize(key): int(value) for key, value in table.get("exemptions", {}).items()}
    return GateConfig(
        max_module_statements=int(table.get("max_module_statements", DEFAULT_MAX_MODULE_STATEMENTS)),
        exemptions=exemptions,
    )


def count_statements(source: str) -> int:
    """Count the AST statement nodes in a module's source.

    Args:
        source: Python source text.

    Returns:
        The number of ``ast.stmt`` nodes, at every nesting depth.

    Raises:
        SyntaxError: If ``source`` does not parse.
    """
    return sum(1 for node in ast.walk(ast.parse(source)) if isinstance(node, ast.stmt))


def measure_package(package_root: UPath) -> dict[str, int]:
    """Count statements for every module in a package tree.

    Args:
        package_root: Directory of the package to walk.

    Returns:
        Mapping of module path to statement count. Paths are relative to
        ``package_root``'s parent, so a key reads ``drevalpy/a/b.py`` whether the
        gate was pointed at ``drevalpy`` or at an absolute path ending in it.

    Raises:
        SystemExit: If ``package_root`` is not a directory, or a module in it
            does not parse.
    """
    if not package_root.is_dir():
        _fail(f"{package_root} is not a directory; run the gate from the repository root.")
    sizes: dict[str, int] = {}
    for path in sorted(package_root.rglob("*.py")):
        key = f"{package_root.name}/{_normalize(str(path.relative_to(package_root)))}"
        try:
            sizes[key] = count_statements(path.read_text())
        except SyntaxError as exc:
            _fail(f"{path} does not parse: {exc}")
    return sizes


def evaluate(sizes: dict[str, int], config: GateConfig) -> list[ModuleSize]:
    """Pair every measured module with the ceiling it is held to.

    Args:
        sizes: Mapping of module path to statement count.
        config: Resolved gate configuration.

    Returns:
        One entry per measured module, sorted by path.
    """
    results = []
    for path in sorted(sizes):
        exempt = path in config.exemptions
        ceiling = config.exemptions[path] if exempt else config.max_module_statements
        results.append(ModuleSize(path=path, statements=sizes[path], ceiling=ceiling, exempt=exempt))
    return results


def find_violations(results: list[ModuleSize]) -> list[ModuleSize]:
    """Select the modules that exceed their effective ceiling.

    Args:
        results: Output of :func:`evaluate`.

    Returns:
        The failing entries, in the order given.
    """
    return [item for item in results if item.statements > item.ceiling]


def find_ratchetable(results: list[ModuleSize], max_module_statements: int) -> list[ModuleSize]:
    """Select exempted modules whose recorded ceiling is now needlessly high.

    Args:
        results: Output of :func:`evaluate`.
        max_module_statements: The global ceiling exemptions are measured against.

    Returns:
        Exempted entries that now clear the global ceiling, or have shrunk well
        below their own recorded ceiling.
    """
    return [
        item
        for item in results
        if item.exempt
        and (item.statements <= max_module_statements or item.statements <= item.ceiling - RATCHET_MARGIN)
    ]


def render_report(results: list[ModuleSize], config: GateConfig) -> tuple[str, bool]:
    """Build the human-readable gate report.

    Args:
        results: Output of :func:`evaluate`.
        config: Resolved gate configuration.

    Returns:
        The report text and whether the gate passed.
    """
    violations = find_violations(results)
    lines: list[str] = []

    ratchetable = find_ratchetable(results, config.max_module_statements)
    if ratchetable:
        lines.append("Size gate: exemptions that can be lowered or deleted")
        lines.extend(
            f"  {item.path}: {item.statements} stmts (recorded ceiling {item.ceiling})" for item in ratchetable
        )
        lines.append("")

    stale = sorted(set(config.exemptions) - {item.path for item in results})
    if stale:
        lines.append("Size gate: exemptions for modules that no longer exist (delete them)")
        lines.extend(f"  {path}" for path in stale)
        lines.append("")

    if not violations:
        lines.append(
            f"Size gate passed: {len(results)} modules at or below their ceiling "
            f"(global {config.max_module_statements} statements, {len(config.exemptions)} exemptions)."
        )
        return "\n".join(lines), True

    lines.append(f"Size gate FAILED: {len(violations)} module(s) above their ceiling.")
    lines.append(f"{'module':<70} {'actual':>8} {'ceiling':>8}")
    lines.extend(f"{item.path:<70} {item.statements:>8} {item.ceiling:>8}" for item in violations)
    lines.append("")
    lines.append(
        "Split the module - a mixin or a `_`-prefixed private helper beside it - or,\n"
        "only if it is genuinely one indivisible unit, record its current statement\n"
        "count in [tool.drevalpy.size_gate].exemptions in pyproject.toml with a reason."
    )
    return "\n".join(lines), False


def main(argv: list[str] | None = None) -> int:
    """Run the size gate.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` if every module meets its ceiling, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description="Enforce a per-module statement ceiling.")
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="Package directory to walk.")
    parser.add_argument("--pyproject", default=DEFAULT_PYPROJECT, help="Path to the pyproject.toml holding the config.")
    args = parser.parse_args(argv)

    config = load_gate_config(UPath(args.pyproject))
    sizes = measure_package(UPath(args.package))
    results = evaluate(sizes, config)
    report, passed = render_report(results, config)
    print(report)
    return 0 if passed else 1


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _fail(message: str) -> NoReturn:
    print(f"size_gate: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

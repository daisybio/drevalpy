"""Enforce a per-module coverage floor on top of coverage.py's global ``fail_under``.

``coverage.py`` can only fail a run on the aggregate percentage, which lets a
single untested module hide behind a well-tested package. This script reads the
``coverage.json`` report and fails if any module falls below its effective
floor.

The floor is ``[tool.drevalpy.coverage_gate].min_file_coverage`` for every
module, except those listed in the ``exemptions`` table, which carry their own
lower floor. Every exemption is technical debt: delete the entry once the module
is properly tested.

Usage::

    uv run pytest -m "not network" --cov --cov-report=json
    uv run python tools/coverage_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from typing import NamedTuple, NoReturn

from upath import UPath

DEFAULT_COVERAGE_JSON = "coverage.json"
DEFAULT_PYPROJECT = "pyproject.toml"
DEFAULT_MIN_FILE_COVERAGE = 50.0
RATCHET_MARGIN = 3.0


class GateConfig(NamedTuple):
    """Resolved ``[tool.drevalpy.coverage_gate]`` settings."""

    min_file_coverage: float
    exemptions: dict[str, float]


class FileCoverage(NamedTuple):
    """One module's measured coverage and the floor it is held to."""

    path: str
    percent: float
    floor: float
    exempt: bool


def load_gate_config(pyproject: UPath) -> GateConfig:
    """Read the coverage-gate configuration from ``pyproject.toml``.

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
    table = data.get("tool", {}).get("drevalpy", {}).get("coverage_gate", {})
    exemptions = {_normalize(key): float(value) for key, value in table.get("exemptions", {}).items()}
    return GateConfig(
        min_file_coverage=float(table.get("min_file_coverage", DEFAULT_MIN_FILE_COVERAGE)),
        exemptions=exemptions,
    )


def load_file_percentages(coverage_json: UPath) -> dict[str, float]:
    """Read per-file coverage percentages from a ``coverage.json`` report.

    Args:
        coverage_json: Path to the JSON report written by ``--cov-report=json``.

    Returns:
        Mapping of normalized module path to percent covered.

    Raises:
        SystemExit: If the report is missing or cannot be parsed.
    """
    if not coverage_json.is_file():
        _fail(
            f"{coverage_json} not found. Produce it first, for example:\n"
            '    uv run pytest -m "not network" --cov --cov-report=json'
        )
    try:
        report = json.loads(coverage_json.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"{coverage_json} is not valid JSON: {exc}")
    return {
        _normalize(path): float(entry["summary"]["percent_covered"]) for path, entry in report.get("files", {}).items()
    }


def evaluate(percentages: dict[str, float], config: GateConfig) -> list[FileCoverage]:
    """Pair every measured module with the floor it is held to.

    Args:
        percentages: Mapping of module path to percent covered.
        config: Resolved gate configuration.

    Returns:
        One entry per measured module, sorted by path.
    """
    results = []
    for path in sorted(percentages):
        exempt = path in config.exemptions
        floor = config.exemptions[path] if exempt else config.min_file_coverage
        results.append(FileCoverage(path=path, percent=percentages[path], floor=floor, exempt=exempt))
    return results


def find_violations(results: list[FileCoverage]) -> list[FileCoverage]:
    """Select the modules that fall below their effective floor.

    Args:
        results: Output of :func:`evaluate`.

    Returns:
        The failing entries, in the order given.
    """
    return [item for item in results if item.percent < item.floor]


def find_ratchetable(results: list[FileCoverage], min_file_coverage: float) -> list[FileCoverage]:
    """Select exempted modules whose recorded floor is now needlessly low.

    Args:
        results: Output of :func:`evaluate`.
        min_file_coverage: The global floor exemptions are measured against.

    Returns:
        Exempted entries that clear the global floor, or sit comfortably above
        their own recorded floor.
    """
    return [
        item
        for item in results
        if item.exempt and (item.percent >= min_file_coverage or item.percent >= item.floor + RATCHET_MARGIN)
    ]


def render_report(results: list[FileCoverage], config: GateConfig) -> tuple[str, bool]:
    """Build the human-readable gate report.

    Args:
        results: Output of :func:`evaluate`.
        config: Resolved gate configuration.

    Returns:
        The report text and whether the gate passed.
    """
    violations = find_violations(results)
    lines: list[str] = []

    ratchetable = find_ratchetable(results, config.min_file_coverage)
    if ratchetable:
        lines.append("Coverage gate: exemptions that can be lowered or deleted")
        lines.extend(f"  {item.path}: {item.percent:.1f}% (recorded floor {item.floor:.0f}%)" for item in ratchetable)
        lines.append("")

    stale = sorted(set(config.exemptions) - {item.path for item in results})
    if stale:
        lines.append("Coverage gate: exemptions for modules absent from the report (delete them)")
        lines.extend(f"  {path}" for path in stale)
        lines.append("")

    if not violations:
        lines.append(
            f"Coverage gate passed: {len(results)} modules at or above their floor "
            f"(global {config.min_file_coverage:.0f}%, {len(config.exemptions)} exemptions)."
        )
        return "\n".join(lines), True

    lines.append(f"Coverage gate FAILED: {len(violations)} module(s) below their floor.")
    lines.append(f"{'module':<70} {'actual':>8} {'floor':>8}")
    lines.extend(f"{item.path:<70} {item.percent:>7.1f}% {item.floor:>7.0f}%" for item in violations)
    lines.append("")
    lines.append(
        "Add tests, or - only if the module is genuinely untestable - record its current\n"
        "floor in [tool.drevalpy.coverage_gate].exemptions in pyproject.toml."
    )
    return "\n".join(lines), False


def main(argv: list[str] | None = None) -> int:
    """Run the coverage gate.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` if every module meets its floor, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description="Enforce a per-module coverage floor.")
    parser.add_argument("--coverage-json", default=DEFAULT_COVERAGE_JSON, help="Path to the coverage JSON report.")
    parser.add_argument("--pyproject", default=DEFAULT_PYPROJECT, help="Path to the pyproject.toml holding the config.")
    args = parser.parse_args(argv)

    config = load_gate_config(UPath(args.pyproject))
    percentages = load_file_percentages(UPath(args.coverage_json))
    results = evaluate(percentages, config)
    report, passed = render_report(results, config)
    print(report)
    return 0 if passed else 1


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _fail(message: str) -> NoReturn:
    print(f"coverage_gate: {message}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

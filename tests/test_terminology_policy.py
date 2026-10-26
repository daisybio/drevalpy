"""Policy guard: "cell line" survives only where the data really are cell lines.

DrEvalPy calls the biological source of a drug response measurement a *sample*
(see GLOSSARY.md); a cell line is one sample type. This test scans the git-tracked
sources under ``src/`` and ``tests/`` for the old wording and fails on any hit that
``ALLOWLIST`` does not cover. Each entry carries a reason, and
``test_allowlist_entries_still_match`` fails once an entry outlives its text.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md")
TERM = re.compile(r"(?i)cell[ _-]?lines?|CellLine")

#: (path glob, regex searched in the line, reason). ``*`` in the glob also matches ``/``.
ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    (
        "src/drevalpy/types/data/dataset/_dataset.py",
        r"cell_line_name|cell-line datasets",
        "documents the optional cellosaurus_id / cell_line_name metadata of cell-line datasets",
    ),
    (
        "src/drevalpy/components/featurizers/sample/gene_lists/_make_gene_lists.py",
        r"cellosaurus_id|cell_line_name",
        "optional columns of cell-line gene-list tables are tolerated, not required",
    ),
    (
        "tests/*",
        r"cell_line_name",
        "tests build and read the optional cell_line_name column of cell-line datasets",
    ),
    (
        "tests/types/data/dataset/test_dataset.py",
        r"cell-line columns",
        "a dataset without the optional cell-line columns (cellosaurus_id / cell_line_name)",
    ),
    (
        "src/drevalpy/components/predictors/literature/srmf/predictor.py",
        r"prediction in cell lines",
        "title of the SRMF paper, which studies cell lines",
    ),
)


@lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Git-tracked files under ``src/`` and ``tests/`` with a scanned suffix."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    try:
        listing = subprocess.run(  # noqa: S603 - fixed argv, git resolved via shutil.which
            [git, "ls-files", "-z", "--", "src", "tests"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git metadata is unavailable")
    paths = (p for p in listing.split("\0") if p.endswith(SCANNED_SUFFIXES) and p != SELF)
    return tuple(p for p in paths if (REPO_ROOT / p).is_file())


@lru_cache(maxsize=1)
def _hits() -> tuple[tuple[str, int, str], ...]:
    """Every ``(path, line number, line)`` that matches ``TERM``."""
    found = []
    for path in _tracked_files():
        text = (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")
        if "cell" not in text.lower():
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if "cell" in line.lower() and TERM.search(line):
                found.append((path, n, line))
    return tuple(found)


def _covers(entry: tuple[str, str, str], path: str, line: str) -> bool:
    glob, pattern, _ = entry
    return fnmatch(path, glob) and re.search(pattern, line) is not None


def test_no_unlisted_cell_line_wording() -> None:
    """Use "sample" in new text; allowlist a hit only if the data really are cell lines."""
    stray = [
        f"{path}:{n}: {line.strip()}" for path, n, line in _hits() if not any(_covers(e, path, line) for e in ALLOWLIST)
    ]
    assert not stray, "'cell line' wording outside the ALLOWLIST (say 'sample', see GLOSSARY.md):\n" + "\n".join(stray)


def test_allowlist_entries_still_match() -> None:
    """An allowlist entry that matches nothing is stale and must be deleted."""
    stale = [
        f"{glob} ~ {pattern}"
        for entry in ALLOWLIST
        for glob, pattern, _ in [entry]
        if not any(_covers(entry, path, line) for path, _, line in _hits())
    ]
    assert not stale, "stale ALLOWLIST entries:\n" + "\n".join(stale)


def test_allowlist_entries_carry_a_reason() -> None:
    """Every entry explains itself."""
    assert all(reason.strip() for _, _, reason in ALLOWLIST)

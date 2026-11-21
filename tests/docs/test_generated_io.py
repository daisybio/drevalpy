"""Tests for docs-only generated include writers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))


def test_write_text_if_changed_skips_identical_content(tmp_path: Path) -> None:
    from _generated_io import write_text_if_changed

    path = tmp_path / "generated.rst"
    assert write_text_if_changed(path, "same\n") is True
    mtime = path.stat().st_mtime_ns
    assert write_text_if_changed(path, "same\n") is False
    assert path.stat().st_mtime_ns == mtime
    assert write_text_if_changed(path, "changed\n") is True
    assert path.read_text(encoding="utf-8") == "changed\n"

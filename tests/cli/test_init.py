"""Tests for the :mod:`drevalpy.cli` package surface."""

from __future__ import annotations

import drevalpy.cli as cli_pkg
from drevalpy.cli import main as cli_main_module


def test_exports_the_app_from_main() -> None:
    assert cli_pkg.app is cli_main_module.app


def test_exports_the_console_script_entry_point() -> None:
    assert cli_pkg.cli_main is cli_main_module.cli_main


def test_all_lists_only_the_two_public_names() -> None:
    assert cli_pkg.__all__ == ["app", "cli_main"]

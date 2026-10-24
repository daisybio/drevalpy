"""Tests for :mod:`drevalpy.cli.catalog._render`: empty cells, row counts and bracketed text not read as markup."""

from __future__ import annotations

from enum import Enum

import pandas as pd
import pytest
from rich.text import Text

from drevalpy.cli.catalog import _render
from tests.cli._helpers import plain


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")


class Flavour(Enum):
    VANILLA = 1


def output(capsys: pytest.CaptureFixture[str]) -> str:
    return plain(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        ("", "-"),
        ("   ", "-"),
        ("  spaced  ", "spaced"),
        (True, "yes"),
        (False, "no"),
        (0, "0"),
        (Flavour.VANILLA, "VANILLA"),
        (frozenset({"b", "a"}), "a, b"),
        (frozenset(), "-"),
        ({"solo"}, "solo"),
        (["first", "second"], "first, second"),
        ((), "-"),
        (frozenset({Flavour.VANILLA}), "VANILLA"),
    ],
)
def test_format_value(value: object, expected: str) -> None:
    assert _render.format_value(value) == expected


def test_render_frame_prints_title_headers_rows_cells_and_count(capsys: pytest.CaptureFixture[str]) -> None:
    frame = pd.DataFrame(
        {
            "Name": ["alpha", "beta"],
            "Description": ["First one", "Second one"],
            "Tags": [frozenset({"baseline"}), frozenset()],
        }
    )

    _render.render_frame(frame, title="Things", empty_hint="nothing")
    printed = output(capsys)

    assert all(text in printed for text in ("Things", "Name", "Description", "Tags", "alpha", "beta", "baseline"))
    assert _render.EMPTY_CELL in printed
    assert "2 entries" in printed

    _render.render_frame(frame.head(1), title="Things", empty_hint="nothing")
    assert "1 entry" in output(capsys)


def test_render_frame_on_an_empty_frame_prints_only_the_hint(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_frame(pd.DataFrame(columns=["Name"]), title="Things", empty_hint="Nothing registered.")
    printed = output(capsys)

    assert "Nothing registered." in printed
    assert "Things" not in printed


def test_render_rows(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_rows(
        [["plugin", Text("loaded", style="green")], ["alpha", "features [bold] and more"]],
        columns=["Plugin", "Status"],
        title="Plugins",
        empty_hint="none",
        show_count=False,
    )
    printed = output(capsys)

    assert "loaded" in printed
    assert "[bold]" in printed
    assert "entries" not in printed

    _render.render_rows([], columns=["Field"], title="Thing", empty_hint="Nothing here.")
    assert "Nothing here." in output(capsys)


def test_render_mapping_prints_pairs_without_a_count_and_hints_when_empty(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_mapping({"name": "alpha", "description": "Does a thing"}, title="alpha")
    printed = output(capsys)

    assert all(text in printed for text in ("alpha", "description", "Does a thing"))
    assert "entries" not in printed

    _render.render_mapping({}, title="alpha")
    assert "No metadata recorded for alpha." in output(capsys)


def test_render_empty_and_console(capsys: pytest.CaptureFixture[str]) -> None:
    _render.render_empty("Nothing at all.")

    assert "Nothing at all." in output(capsys)
    assert _render.console() is not _render.console()

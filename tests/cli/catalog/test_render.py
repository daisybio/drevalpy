"""Tests for :mod:`drevalpy.cli.catalog._render`, the rich rendering helpers.

These exercise the helpers directly rather than through a command, because the
behaviour worth pinning is formatting: what an empty cell looks like, that a row
count is reported, and that author-supplied text containing square brackets is
not eaten as rich markup.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
import pytest
from rich.text import Text

from drevalpy.cli.catalog import _render
from tests.cli._helpers import plain


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the console width so wrapping cannot split the asserted text.

    Args:
        monkeypatch: Fixture used to set the environment rich reads.
    """
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")


class Flavour(Enum):
    """Enum stand-in for the enum members registry metadata carries."""

    VANILLA = 1


def output(capsys: pytest.CaptureFixture[str]) -> str:
    """Return the captured stdout with escape codes removed.

    Args:
        capsys: Fixture holding the captured streams.

    Returns:
        Plain-text stdout.
    """
    return plain(capsys.readouterr().out)


class TestFormatValue:
    """``format_value`` turns registry values into display text."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(None, "-", id="none"),
            pytest.param("", "-", id="empty-string"),
            pytest.param("   ", "-", id="whitespace-only"),
            pytest.param("  spaced  ", "spaced", id="stripped"),
            pytest.param(True, "yes", id="true"),
            pytest.param(False, "no", id="false"),
            pytest.param(0, "0", id="zero-is-not-empty"),
            pytest.param(Flavour.VANILLA, "VANILLA", id="enum-member-name"),
            pytest.param(frozenset({"b", "a"}), "a, b", id="frozenset-sorted"),
            pytest.param(frozenset(), "-", id="empty-frozenset"),
            pytest.param({"solo"}, "solo", id="set"),
            pytest.param(["first", "second"], "first, second", id="list-keeps-order"),
            pytest.param((), "-", id="empty-tuple"),
        ],
    )
    def test_formatting(self, value: object, expected: str) -> None:
        assert _render.format_value(value) == expected

    def test_nested_enum_inside_a_frozenset(self) -> None:
        assert _render.format_value(frozenset({Flavour.VANILLA})) == "VANILLA"


class TestRenderFrame:
    """``render_frame`` prints a registry ``table()`` DataFrame."""

    @pytest.fixture()
    def frame(self) -> pd.DataFrame:
        """A two-row frame shaped like a registry table."""
        return pd.DataFrame(
            {
                "Name": ["alpha", "beta"],
                "Description": ["First one", "Second one"],
                "Tags": [frozenset({"baseline"}), frozenset()],
            }
        )

    def test_prints_the_title(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame, title="Things", empty_hint="nothing")

        assert "Things" in output(capsys)

    def test_prints_the_column_headers(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame, title="Things", empty_hint="nothing")
        printed = output(capsys)

        assert "Name" in printed
        assert "Description" in printed
        assert "Tags" in printed

    def test_prints_every_row(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame, title="Things", empty_hint="nothing")
        printed = output(capsys)

        assert "alpha" in printed
        assert "beta" in printed

    def test_formats_cell_values(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame, title="Things", empty_hint="nothing")
        printed = output(capsys)

        assert "baseline" in printed
        assert _render.EMPTY_CELL in printed

    def test_reports_the_row_count(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame, title="Things", empty_hint="nothing")

        assert "2 entries" in output(capsys)

    def test_row_count_is_singular_for_one_row(self, frame: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(frame.head(1), title="Things", empty_hint="nothing")

        assert "1 entry" in output(capsys)

    def test_empty_frame_prints_the_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(pd.DataFrame(columns=["Name"]), title="Things", empty_hint="Nothing registered.")

        assert "Nothing registered." in output(capsys)

    def test_empty_frame_prints_no_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_frame(pd.DataFrame(columns=["Name"]), title="Things", empty_hint="Nothing registered.")

        assert "Things" not in output(capsys)


class TestRenderRows:
    """``render_rows`` is the shared primitive for pre-formatted tables."""

    def test_styled_cells_render_their_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_rows(
            [["plugin", Text("loaded", style="green")]],
            columns=["Plugin", "Status"],
            title="Plugins",
            empty_hint="none",
        )

        assert "loaded" in output(capsys)

    def test_square_brackets_are_not_read_as_markup(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A description like ``[bold]`` must survive verbatim, not vanish."""
        _render.render_rows(
            [["alpha", "features [bold] and more"]],
            columns=["Name", "Description"],
            title="Things",
            empty_hint="none",
        )

        assert "[bold]" in output(capsys)

    def test_count_can_be_suppressed(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_rows(
            [["a", "b"]],
            columns=["Field", "Value"],
            title="Thing",
            empty_hint="none",
            show_count=False,
        )

        assert "1 entry" not in output(capsys)

    def test_no_rows_prints_the_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_rows([], columns=["Field"], title="Thing", empty_hint="Nothing here.")

        assert "Nothing here." in output(capsys)


class TestRenderMapping:
    """``render_mapping`` prints a metadata dict as field/value pairs."""

    def test_prints_the_name_as_the_title(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_mapping({"name": "alpha"}, title="alpha")

        assert "alpha" in output(capsys)

    def test_prints_field_and_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_mapping({"description": "Does a thing"}, title="alpha")
        printed = output(capsys)

        assert "description" in printed
        assert "Does a thing" in printed

    def test_does_not_report_a_field_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A field count is noise: the fields are fixed by the registry, not discovered."""
        _render.render_mapping({"name": "alpha", "description": "Does a thing"}, title="alpha")

        assert "entries" not in output(capsys)

    def test_empty_mapping_prints_the_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_mapping({}, title="alpha")

        assert "No metadata recorded for alpha." in output(capsys)


class TestRenderEmpty:
    """``render_empty`` is the shared not-found path."""

    def test_prints_the_hint(self, capsys: pytest.CaptureFixture[str]) -> None:
        _render.render_empty("Nothing at all.")

        assert "Nothing at all." in output(capsys)


class TestConsole:
    """The console is built per call, not cached at import time."""

    def test_returns_a_fresh_console_each_time(self) -> None:
        assert _render.console() is not _render.console()

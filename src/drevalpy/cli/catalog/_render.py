"""Rich rendering helpers shared by the ``drevalpy list`` commands.

Every cell is handed to rich as a :class:`~rich.text.Text` instance rather than a
markup string: registry descriptions are free-form author text and a stray
``[...]`` in one of them would otherwise be swallowed as a style tag.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

#: Placeholder for an empty cell, so a blank column reads as "nothing here"
#: rather than as a rendering bug.
EMPTY_CELL = "-"


def console() -> Console:
    """Build a console for one render pass.

    Constructed per call rather than at import time because click's
    ``CliRunner`` swaps ``sys.stdout`` for each invocation, and because terminal
    width is then measured when the output is actually produced.

    Returns:
        A fresh :class:`~rich.console.Console`.
    """
    from rich.console import Console

    return Console()


def format_value(value: Any) -> str:
    """Render one registry value as display text.

    Args:
        value: Value taken from a registry table cell or metadata mapping.

    Returns:
        Human-readable text: collections become comma-separated, enums become
        their member name, booleans become ``yes``/``no``, and anything empty
        becomes :data:`EMPTY_CELL`.
    """
    if value is None:
        return EMPTY_CELL
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Enum):
        return str(value.name)
    if isinstance(value, frozenset | set):
        return ", ".join(sorted(format_value(item) for item in value)) or EMPTY_CELL
    if isinstance(value, list | tuple):
        return ", ".join(format_value(item) for item in value) or EMPTY_CELL
    return str(value).strip() or EMPTY_CELL


def _text(cell: str | Text) -> Text:
    """Coerce a cell to rich text with markup disabled.

    Args:
        cell: Either display text or an already-styled ``Text``.

    Returns:
        The ``Text`` to hand to rich.
    """
    from rich.text import Text

    return cell if isinstance(cell, Text) else Text(cell)


def _table(title: str, columns: Iterable[str], caption: str | None = None) -> Table:
    """Build an empty table with folding columns.

    Args:
        title: Heading printed above the table.
        columns: Column headers, in order.
        caption: Optional line printed below the table.

    Returns:
        A table ready for :meth:`~rich.table.Table.add_row`.
    """
    from rich import box
    from rich.table import Table

    table = Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAD,
        title_justify="left",
        caption_justify="left",
        header_style="bold",
        pad_edge=False,
    )
    for column in columns:
        # Long descriptions wrap inside their column instead of being truncated
        # or pushing the table past the terminal width.
        table.add_column(column, overflow="fold")
    return table


def render_empty(hint: str) -> None:
    """Print the stand-in shown when there is nothing to list.

    Args:
        hint: Sentence explaining what was empty and, ideally, why.
    """
    from rich.text import Text

    console().print(Text(hint, style="yellow"))


def render_rows(
    rows: Sequence[Sequence[str | Text]],
    *,
    columns: Sequence[str],
    title: str,
    empty_hint: str,
    show_count: bool = True,
) -> None:
    """Print pre-formatted rows as a table.

    Args:
        rows: One sequence of cells per row, matching ``columns`` in length.
        columns: Column headers, in order.
        title: Heading printed above the table.
        empty_hint: Printed instead of the table when ``rows`` is empty.
        show_count: Add a row-count caption below the table.
    """
    if not rows:
        render_empty(empty_hint)
        return
    table = _table(title, columns, caption=_count_caption(len(rows)) if show_count else None)
    for row in rows:
        table.add_row(*(_text(cell) for cell in row))
    console().print(table)


def render_frame(frame: pd.DataFrame, *, title: str, empty_hint: str) -> None:
    """Print a registry ``table()`` DataFrame.

    Args:
        frame: DataFrame as returned by any registry's ``table()``.
        title: Heading printed above the table.
        empty_hint: Printed instead of the table when the frame has no rows.
    """
    if frame.empty:
        render_empty(empty_hint)
        return
    rows = [[format_value(value) for value in row] for row in frame.itertuples(index=False)]
    render_rows(rows, columns=[str(column) for column in frame.columns], title=title, empty_hint=empty_hint)


def render_mapping(data: Mapping[str, Any], *, title: str) -> None:
    """Print a metadata mapping as a two-column field/value table.

    Args:
        data: Mapping as returned by any registry's ``metadata()``.
        title: Heading printed above the table, normally the entry's name.
    """
    rows = [[str(key), format_value(value)] for key, value in data.items()]
    render_rows(
        rows,
        columns=["Field", "Value"],
        title=title,
        empty_hint=f"No metadata recorded for {title}.",
        show_count=False,
    )


def _count_caption(count: int) -> str:
    """Return the row-count caption for a table.

    Args:
        count: Number of rows rendered.

    Returns:
        Caption text with the count correctly pluralised.
    """
    return f"{count} entry" if count == 1 else f"{count} entries"

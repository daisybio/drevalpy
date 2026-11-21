"""Helpers for writing generated Sphinx includes without spurious rebuilds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def write_text_if_changed(path: Path, text: str, *, encoding: str = "utf-8") -> bool:
    """Write ``text`` to ``path`` only when content differs.

    Leaving the mtime unchanged when content is identical prevents
    ``sphinx-autobuild`` from entering a rebuild loop on generated includes.

    :param path: destination file
    :param text: full file contents to write
    :param encoding: text encoding
    :returns: ``True`` if the file was created or updated
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding=encoding) == text:
        return False
    path.write_text(text, encoding=encoding)
    return True


def list_table(widths: Sequence[int], header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Render a ``.. list-table::`` directive with a header row.

    :param widths: relative column widths
    :param header: column titles
    :param rows: table rows, each a sequence of cell texts as long as *header*
    :returns: the directive lines, without a trailing blank line
    """
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {' '.join(map(str, widths))}", ""]
    for row in [header, *rows]:
        lines.extend(f"   {'*' if column == 0 else ' '} - {cell}" for column, cell in enumerate(row))
    return lines


def registry_description(registry: Any, name: str) -> str:
    """Return the one-line description a component registered, for a generated table cell.

    :param registry: component registry that holds *name*
    :param name: registry name
    :returns: the description with whitespace collapsed
    :raises RuntimeError: if the component has no description
    """
    description = " ".join((registry.metadata(name).get("description") or "").split())
    if not description:
        raise RuntimeError(f"Component {name!r} is missing a description")
    return description

"""Helpers for writing generated Sphinx includes without spurious rebuilds."""

from __future__ import annotations

from pathlib import Path


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

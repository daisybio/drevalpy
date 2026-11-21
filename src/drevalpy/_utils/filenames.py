"""Turn model names into file and directory names that every operating system accepts."""

from __future__ import annotations

import re
from collections.abc import Iterable

# Windows rejects these in any path component; ':' appears in every featurizer:predictor spec.
_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str) -> str:
    """Replace characters Windows forbids in file names with ``_``.

    Trailing dots and spaces are stripped too, since Windows drops them silently.
    The result is the same on every platform, so result trees stay portable.

    :param name: Model or spec name, e.g. ``pca[gene_expression]:fingerprints:ridge``.
    :returns: A non-empty name usable as a single path component.
    """
    return _FORBIDDEN.sub("_", name).rstrip(". ") or "_"


def unique_safe_filenames(names: Iterable[str]) -> dict[str, str]:
    """Map each name to a :func:`safe_filename` no other name in *names* maps to.

    Names that sanitize to the same string, compared case-insensitively as on Windows
    and macOS, get a ``-2``, ``-3``, ... suffix in order.

    :param names: Names in the order their files are written.
    :returns: ``name -> file name``.
    """
    result: dict[str, str] = {}
    taken: set[str] = set()
    for name in names:
        base = candidate = safe_filename(name)
        n = 1
        while candidate.lower() in taken:
            n += 1
            candidate = f"{base}-{n}"
        taken.add(candidate.lower())
        result[name] = candidate
    return result

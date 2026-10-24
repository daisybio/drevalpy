"""Turn model names into file and directory names that every operating system accepts."""

from __future__ import annotations

import re

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

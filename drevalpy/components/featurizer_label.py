"""Stable labels for featurizer configs in concat blocks and state."""

from __future__ import annotations

from drevalpy.components.view_aliases import format_view_alias

_VIEW_PARAMETRIC_FEATURIZERS = frozenset({"raw", "pca"})


def featurizer_block_label(name: str, view: str | None = None, *, occurrence: int = 0) -> str:
    """Return a stable block label for concat outputs and saved state."""
    base = f"{name}[{format_view_alias(view)}]" if view is not None else name
    if occurrence > 0:
        return f"{base}#{occurrence}"
    return base


def featurizer_config_block_label(name: str, view: str | None, *, occurrence: int = 0) -> str:
    """Return the concat block label for a normalized featurizer config."""
    return featurizer_block_label(name, view, occurrence=occurrence)


def requires_explicit_view(name: str) -> bool:
    """Return whether a featurizer registry name requires an explicit view."""
    return name in _VIEW_PARAMETRIC_FEATURIZERS

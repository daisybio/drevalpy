"""Stable labels for featurizer configs in concat blocks and HPO keys."""

from __future__ import annotations

from drevalpy.components.view_aliases import format_view_alias

_VIEW_PARAMETRIC_FEATURIZERS = frozenset({"raw", "pca"})


def qualified_featurizer_selector(name: str, view: str | None = None) -> str:
    """Return the canonical selector for a featurizer leaf.

    View-specific featurizers use bracket syntax (``pca[expression]``). Non-view
    featurizers use the bare registry name (``landmarkGenes``).
    """
    if view is not None:
        return f"{name}[{format_view_alias(view)}]"
    return name


def featurizer_block_label(name: str, view: str | None = None) -> str:
    """Return a stable block label for concat outputs and saved state."""
    return qualified_featurizer_selector(name, view)


def featurizer_config_block_label(name: str, view: str | None) -> str:
    """Return the concat block label for a normalized featurizer config."""
    return qualified_featurizer_selector(name, view)


def requires_explicit_view(name: str) -> bool:
    """Return whether a featurizer registry name requires an explicit view."""
    return name in _VIEW_PARAMETRIC_FEATURIZERS

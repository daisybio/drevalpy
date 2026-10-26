"""Stable labels for featurizer configs in concat blocks and HPO keys."""

from __future__ import annotations

from drevalpy.registry.sample_featurizer import get as get_sample_featurizer


def qualified_featurizer_selector(name: str, view: str | None = None) -> str:
    """Return the canonical selector for a featurizer leaf.

    View-specific featurizers use bracket syntax (``pca[expression]``). Non-view
    featurizers use the bare registry name (``landmarkGenes``).

    :param name: Featurizer registry name.
    :param view: Optional explicit omics view.
    :returns: Canonical selector string for HPO keys and concat blocks.
    """
    if view is not None:
        return f"{name}[{view}]"
    return name


def featurizer_config_block_label(name: str, view: str | None) -> str:
    """Return the concat block label for a normalized featurizer config.

    :param name: Featurizer registry name.
    :param view: Explicit omics view, if the config sets one.
    :returns: Qualified selector used as the concat block label.
    """
    return qualified_featurizer_selector(name, view)


def requires_explicit_view(name: str) -> bool:
    """Return whether a featurizer registry name requires an explicit view.

    :param name: Featurizer registry name, looked up in the sample featurizer registry.
    :returns: ``True`` when the class sets ``requires_view``; ``False`` for unregistered names.
    """
    try:
        cls = get_sample_featurizer(name)
    except (ValueError, ImportError):
        return False
    return bool(getattr(cls, "requires_view", False))

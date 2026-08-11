"""Walk and validate featurizer config trees."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig


def iter_featurizer_leaves(
    featurizer: FeaturizerConfig,
    registry: str,
) -> Iterator[FeaturizerConfig]:
    """Yield leaf featurizer configs from a tree (concat parents are expanded).

    :param featurizer: Root featurizer config, possibly a concat parent.
    :param registry: Default registry used when normalizing nested children.
    :yields: Leaf ``FeaturizerConfig`` nodes.
    """
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.featurizers or ():
            yield from iter_featurizer_leaves(child, registry)
        return
    yield featurizer


def ensure_unique_qualified_featurizers(featurizer: FeaturizerConfig, registry: str) -> None:
    """Raise ``ValueError`` when a registry slot repeats a qualified selector.

    Duplicate means the same qualified selector (for example ``raw[expression]``)
    appears more than once under one registry. The same base name on different
    views (``raw[expression]+raw[mutations]``) is allowed.

    :param featurizer: Featurizer tree to validate (concat parents are walked).
    :param registry: Registry slot name used in error messages.
    :raises ValueError: If the same qualified selector appears twice.
    """
    if featurizer.name != "concatFeaturizers":
        return
    seen: set[str] = set()
    for leaf in iter_featurizer_leaves(featurizer, registry):
        from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector

        selector = qualified_featurizer_selector(leaf.name, leaf.view)
        if selector in seen:
            msg = (
                f"Duplicate featurizer selector {selector!r} in registry {registry!r}. "
                "Each qualified featurizer may appear at most once per slot."
            )
            raise ValueError(msg)
        seen.add(selector)

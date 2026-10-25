"""Resolve view names from featurizer configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.registry.featurizer import Side, get_featurizer

if TYPE_CHECKING:
    from drevalpy.models.config.resolved import ResolvedModelConfig


def entity_id_only_from_featurizer_config(config: FeaturizerConfig, *, registry: Side) -> bool:
    """Return True when the featurizer only needs entity identifiers, not omics or drug views.

    :param config: Featurizer config node to inspect.
    :param registry: ``sample`` or ``drug`` registry label.
    :returns: ``True`` when the featurizer tree is entity-id-only.
    """
    if config.name == "concatFeaturizers":
        children = config.featurizers or ()
        if not children:
            return False
        return all(entity_id_only_from_featurizer_config(child, registry=registry) for child in children)
    return bool(getattr(get_featurizer(registry, config.name), "entity_id_only", False))


def _concat_child_views(
    config: FeaturizerConfig,
    *,
    registry: Side,
    resolved: ResolvedModelConfig | None,
) -> list[str]:
    views: list[str] = []
    for child in config.featurizers or ():
        views.extend(views_from_featurizer_config(child, registry=registry, resolved=resolved))
    return views


def views_from_featurizer_config(
    config: FeaturizerConfig,
    *,
    registry: Side,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Ask each featurizer in the tree which raw views it reads.

    :param config: Featurizer config node, possibly a concat parent.
    :param registry: ``sample`` or ``drug`` registry label.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Raw view names required by the featurizer tree.
    """
    if config.name == "concatFeaturizers":
        return _concat_child_views(config, registry=registry, resolved=resolved)
    cls = get_featurizer(registry, config.name)
    kwargs = featurizer_leaf_kwargs(config, registry=registry, resolved=resolved)
    return list(cls.resolve_input_views(**kwargs))

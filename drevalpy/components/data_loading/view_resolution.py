"""Resolve legacy view names from featurizer configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
    from drevalpy.models.config.model import ModelConfig
    from drevalpy.models.config.resolved import ResolvedModelConfig


def _featurizer_cls(config: FeaturizerConfig, *, registry: str) -> type[Any]:
    from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

    if registry == "cell_line":
        return get_cell_line_featurizer(config.name)
    return get_drug_featurizer(config.name)


def entity_id_only_from_featurizer_config(config: FeaturizerConfig, *, registry: str) -> bool:
    """Return True when the featurizer only needs entity identifiers, not omics or drug views.

    :param config: Featurizer config node to inspect.
    :param registry: ``cell_line`` or ``drug`` registry label.
    :returns: ``True`` when the featurizer tree is entity-id-only.
    """
    if config.name == "concatFeaturizers":
        children = config.featurizers or ()
        if not children:
            return False
        return all(entity_id_only_from_featurizer_config(child, registry=registry) for child in children)
    return bool(getattr(_featurizer_cls(config, registry=registry), "entity_id_only", False))


def cell_line_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured cell-line featurizer only needs entity ids.

    :param config: Model configuration to inspect.
    :returns: ``True`` when no cell-line omics views are required.
    """
    if config.cell_line_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.cell_line_featurizer, registry="cell_line")


def drug_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured drug featurizer only needs entity ids.

    :param config: Model configuration to inspect.
    :returns: ``True`` when no drug feature views are required.
    """
    if config.drug_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.drug_featurizer, registry="drug")


def _concat_child_views(
    config: FeaturizerConfig,
    *,
    registry: Literal["cell_line", "drug"],
    resolved: ResolvedModelConfig | None,
) -> list[str]:
    views: list[str] = []
    for child in config.featurizers or ():
        views.extend(views_from_featurizer_config(child, registry=registry, resolved=resolved))
    return views


def views_from_featurizer_config(
    config: FeaturizerConfig,
    *,
    registry: Literal["cell_line", "drug"],
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Ask each featurizer in the tree which raw views it reads.

    :param config: Featurizer config node, possibly a concat parent.
    :param registry: ``cell_line`` or ``drug`` registry label.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Raw view names required by the featurizer tree.
    """
    from drevalpy.components.data_loading.leaf_kwargs import featurizer_leaf_kwargs

    if config.name == "concatFeaturizers":
        return _concat_child_views(config, registry=registry, resolved=resolved)
    cls = _featurizer_cls(config, registry=registry)
    kwargs = featurizer_leaf_kwargs(config, registry=registry, resolved=resolved)
    return list(cls.resolve_input_views(**kwargs))


def cell_line_views_from_model_config(
    config: ModelConfig,
    *,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Resolve legacy cell-line view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Legacy cell-line view names required by the config.
    """
    if config.cell_line_featurizer is None:
        return []
    return views_from_featurizer_config(
        config.cell_line_featurizer,
        registry="cell_line",
        resolved=resolved,
    )


def drug_views_from_model_config(
    config: ModelConfig,
    *,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Resolve legacy drug view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Legacy drug view names required by the config.
    """
    if config.drug_featurizer is None:
        return []
    return views_from_featurizer_config(
        config.drug_featurizer,
        registry="drug",
        resolved=resolved,
    )


def cell_line_views_from_resolved(resolved: ResolvedModelConfig) -> list[str]:
    """Resolve cell-line views from a resolved instance config.

    :param resolved: Resolved model configuration.
    :returns: Legacy cell-line view names.
    """
    return cell_line_views_from_model_config(resolved.template, resolved=resolved)


def drug_views_from_resolved(resolved: ResolvedModelConfig) -> list[str]:
    """Resolve drug views from a resolved instance config.

    :param resolved: Resolved model configuration.
    :returns: Legacy drug view names.
    """
    return drug_views_from_model_config(resolved.template, resolved=resolved)

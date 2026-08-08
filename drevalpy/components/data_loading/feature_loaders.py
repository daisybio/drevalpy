"""Feature loaders for component-based models."""

from __future__ import annotations

from typing import Literal

from drevalpy.components.data_loading.leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.components.data_loading.view_resolution import views_from_featurizer_config
from drevalpy.components.featurizer_tree import iter_featurizer_leaves
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import (
    load_cl_ids_and_tissues_from_csv,
    load_cl_ids_from_csv,
    load_drug_ids_from_csv,
    load_tissues_from_csv,
)
from drevalpy.datasets.loading.views import load_cell_line_feature_views, load_drug_feature_views
from drevalpy.models.config import FeaturizerConfig, ModelConfig, ResolvedModelConfig


def _merge_features(current: FeatureDataset | None, incoming: FeatureDataset) -> FeatureDataset:
    """Merge distinct feature views while rejecting ambiguous collisions.

    :param current: Existing feature dataset, or ``None`` when this is the first view.
    :param incoming: New views to add to *current*.

    :returns: Combined ``FeatureDataset`` with views from both inputs.

    :raises ValueError: If *incoming* reuses a view name already present in *current*.
    """
    if current is None:
        return incoming
    overlap = set(current.view_names).intersection(incoming.view_names)
    if overlap:
        raise ValueError(f"Featurizer loaders emitted duplicate views: {sorted(overlap)}")
    current.add_features(incoming)
    return current


def _has_custom_loader(featurizer_cls: type[Featurizer]) -> bool:
    """Return whether a featurizer overrides the optional disk-loading hook.

    :param featurizer_cls: Featurizer class to inspect.

    :returns: ``True`` when ``load_features`` is overridden on the subclass.
    """
    return getattr(featurizer_cls.load_features, "__func__", None) is not getattr(
        Featurizer.load_features, "__func__", None
    )


def _unwrap_model_config(config: ModelConfig | ResolvedModelConfig) -> tuple[ModelConfig, ResolvedModelConfig | None]:
    if isinstance(config, ResolvedModelConfig):
        return config.template, config
    return config, None


def _load_from_featurizer_tree(
    config: FeaturizerConfig,
    *,
    registry: Literal["cell_line", "drug"],
    dataset_name: str,
    resolved: ResolvedModelConfig | None = None,
) -> FeatureDataset | None:
    """Load every leaf's raw data, using bespoke loaders where available.

    :param config: Featurizer tree configuration for the requested registry.
    :param registry: Whether to resolve cell-line or drug featurizers.
    :param dataset_name: Dataset subdirectory or registry name.
    :param resolved: Optional resolved instance values for tunable loader kwargs.

    :returns: Merged ``FeatureDataset`` for all leaves, or ``None`` when nothing was loaded.
    """
    loaded: FeatureDataset | None = None
    for leaf in iter_featurizer_leaves(config, registry):
        cls = get_cell_line_featurizer(leaf.name) if registry == "cell_line" else get_drug_featurizer(leaf.name)
        kwargs = featurizer_leaf_kwargs(leaf, registry=registry, resolved=resolved)
        if _has_custom_loader(cls):
            loaded = _merge_features(loaded, cls.load_features(dataset_name, **kwargs))
            continue
        views = views_from_featurizer_config(leaf, registry=registry, resolved=resolved)
        if not views:
            continue
        fallback = (
            load_cell_line_feature_views(views, dataset_name)
            if registry == "cell_line"
            else load_drug_feature_views(views, dataset_name)
        )
        if fallback is not None:
            loaded = _merge_features(loaded, fallback)
    return loaded


def load_cell_line_features_for_model_config(
    config: ModelConfig | ResolvedModelConfig,
    dataset_name: str,
) -> FeatureDataset:
    """Load cell-line features implied by *config*, including identity-only featurizers.

    :param config: Template or resolved model configuration.
    :param dataset_name: Dataset subdirectory or registry name.

    :returns: ``FeatureDataset`` with views required by the cell-line featurizer tree.
    """
    template, resolved = _unwrap_model_config(config)
    featurizer = template.cell_line_featurizer
    if featurizer is not None and featurizer.name == "tissue":
        return load_tissues_from_csv(dataset_name)
    if template.predictor.name == "naiveMeanEffects" and (featurizer is None or featurizer.name == "identity"):
        return load_cl_ids_and_tissues_from_csv(dataset_name)
    if template.cell_line_entity_id_only():
        return load_cl_ids_from_csv(dataset_name)
    if featurizer is None:
        return load_cl_ids_from_csv(dataset_name)
    loaded = _load_from_featurizer_tree(
        featurizer,
        registry="cell_line",
        dataset_name=dataset_name,
        resolved=resolved,
    )
    return loaded if loaded is not None else load_cl_ids_from_csv(dataset_name)


def load_drug_features_for_model_config(
    config: ModelConfig | ResolvedModelConfig,
    dataset_name: str,
) -> FeatureDataset | None:
    """Load drug features implied by *config*, including identity-only featurizers.

    :param config: Template or resolved model configuration.
    :param dataset_name: Dataset subdirectory or registry name.

    :returns: ``FeatureDataset`` with drug views, or ``None`` when the model has no drug featurizer.
    """
    template, resolved = _unwrap_model_config(config)
    if template.drug_featurizer is None:
        return None
    if template.drug_entity_id_only():
        return load_drug_ids_from_csv(dataset_name)
    return _load_from_featurizer_tree(
        template.drug_featurizer,
        registry="drug",
        dataset_name=dataset_name,
        resolved=resolved,
    )

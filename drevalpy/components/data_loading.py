"""Load raw feature tables for component-based models."""

from __future__ import annotations

from typing import Literal

from drevalpy.components.featurizer_tree import iter_featurizer_leaves
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer
from drevalpy.data.features import (
    load_cl_ids_and_tissues_from_csv,
    load_cl_ids_from_csv,
    load_drug_ids_from_csv,
    load_multi_cell_line_view,
    load_single_cell_line_view,
    load_single_drug_view,
    load_tissues_from_csv,
)
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models.config import FeaturizerConfig, ModelConfig
from drevalpy.models.featurizer_mapping import (
    _views_from_featurizer_config,
    cell_line_entity_id_only_from_model_config,
    drug_entity_id_only_from_model_config,
)


def load_cell_line_feature_views(
    views: list[str],
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset:
    """Load cell-line features for the configured cell-line views."""
    if len(views) == 1:
        return load_single_cell_line_view(views, data_path, dataset_name, model_name)
    return load_multi_cell_line_view(views, data_path, dataset_name, model_name)


def load_drug_feature_views(
    views: list[str],
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset | None:
    """Load drug features for the configured drug views."""
    if not views:
        return None
    return load_single_drug_view(views, data_path, dataset_name, model_name)


def load_tissue_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """Load tissue labels keyed by cell line id."""
    return load_tissues_from_csv(data_path, dataset_name)


def _merge_features(current: FeatureDataset | None, incoming: FeatureDataset) -> FeatureDataset:
    """Merge distinct feature views while rejecting ambiguous collisions."""
    if current is None:
        return incoming
    overlap = set(current.view_names).intersection(incoming.view_names)
    if overlap:
        raise ValueError(f"Featurizer loaders emitted duplicate views: {sorted(overlap)}")
    current.add_features(incoming)
    return current


def _has_custom_loader(featurizer_cls: type[Featurizer]) -> bool:
    """Return whether a featurizer overrides the optional disk-loading hook."""
    return getattr(featurizer_cls.load_features, "__func__", None) is not getattr(
        Featurizer.load_features, "__func__", None
    )


def _load_from_featurizer_tree(
    config: FeaturizerConfig,
    *,
    registry: Literal["cell_line", "drug"],
    data_path: str,
    dataset_name: str,
    model_name: str,
) -> FeatureDataset | None:
    """Load every leaf's raw data, using bespoke loaders where available."""
    loaded: FeatureDataset | None = None
    for leaf in iter_featurizer_leaves(config, registry):
        cls = get_cell_line_featurizer(leaf.name) if registry == "cell_line" else get_drug_featurizer(leaf.name)
        kwargs = dict(leaf.hyperparameters)
        if leaf.view is not None:
            kwargs.setdefault("view", leaf.view)
        if _has_custom_loader(cls):
            loaded = _merge_features(loaded, cls.load_features(data_path, dataset_name, **kwargs))
            continue
        views = _views_from_featurizer_config(leaf, registry=registry)
        if not views:
            continue
        fallback = (
            load_cell_line_feature_views(views, data_path, dataset_name, model_name=model_name)
            if registry == "cell_line"
            else load_drug_feature_views(views, data_path, dataset_name, model_name=model_name)
        )
        if fallback is not None:
            loaded = _merge_features(loaded, fallback)
    return loaded


def load_cell_line_id_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """Load cell-line identifier features."""
    return load_cl_ids_from_csv(data_path, dataset_name)


def load_cell_line_features_for_model_config(
    config: ModelConfig,
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset:
    """Load cell-line features implied by *config*, including identity-only featurizers."""
    featurizer = config.cell_line_featurizer
    if featurizer is not None and featurizer.name == "tissue":
        return load_tissues_from_csv(data_path, dataset_name)
    if config.predictor.name == "naiveMeanEffects" and (featurizer is None or featurizer.name == "identity"):
        return load_cl_ids_and_tissues_from_csv(data_path, dataset_name)
    if cell_line_entity_id_only_from_model_config(config):
        return load_cl_ids_from_csv(data_path, dataset_name)
    if featurizer is None:
        return load_cl_ids_from_csv(data_path, dataset_name)
    loaded = _load_from_featurizer_tree(
        featurizer,
        registry="cell_line",
        data_path=data_path,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return loaded if loaded is not None else load_cl_ids_from_csv(data_path, dataset_name)


def load_drug_features_for_model_config(
    config: ModelConfig,
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset | None:
    """Load drug features implied by *config*, including identity-only featurizers."""
    if config.drug_featurizer is None:
        return None
    if drug_entity_id_only_from_model_config(config):
        return load_drug_ids_from_csv(data_path, dataset_name)
    return _load_from_featurizer_tree(
        config.drug_featurizer,
        registry="drug",
        data_path=data_path,
        dataset_name=dataset_name,
        model_name=model_name,
    )

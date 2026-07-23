"""Load raw feature tables for component-based models."""

from __future__ import annotations

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
from drevalpy.models.config import ModelConfig
from drevalpy.models.featurizer_mapping import (
    cell_line_entity_id_only_from_model_config,
    cell_line_views_from_model_config,
    drug_entity_id_only_from_model_config,
    drug_views_from_model_config,
)


def load_cell_line_feature_views(
    views: list[str],
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "ComposedModel",
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
    model_name: str = "ComposedModel",
) -> FeatureDataset | None:
    """Load drug features for the configured drug views."""
    if not views:
        return None
    return load_single_drug_view(views, data_path, dataset_name, model_name)


def load_tissue_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """Load tissue labels keyed by cell line id."""
    return load_tissues_from_csv(data_path, dataset_name)


def load_cell_line_id_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """Load cell-line identifier features."""
    return load_cl_ids_from_csv(data_path, dataset_name)


def load_cell_line_features_for_model_config(
    config: ModelConfig,
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "ComposedModel",
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
    views = cell_line_views_from_model_config(config)
    if not views:
        return load_cl_ids_from_csv(data_path, dataset_name)
    return load_cell_line_feature_views(views, data_path, dataset_name, model_name=model_name)


def load_drug_features_for_model_config(
    config: ModelConfig,
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "ComposedModel",
) -> FeatureDataset | None:
    """Load drug features implied by *config*, including identity-only featurizers."""
    if config.drug_featurizer is None:
        return None
    if drug_entity_id_only_from_model_config(config):
        return load_drug_ids_from_csv(data_path, dataset_name)
    views = drug_views_from_model_config(config)
    return load_drug_feature_views(views, data_path, dataset_name, model_name=model_name)

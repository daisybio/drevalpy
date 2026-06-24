"""Load raw feature tables for component-based models."""

from __future__ import annotations

from drevalpy.data.features import (
    load_cl_ids_from_csv,
    load_multi_cell_line_view,
    load_single_cell_line_view,
    load_single_drug_view,
    load_tissues_from_csv,
)
from drevalpy.datasets.dataset import FeatureDataset


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

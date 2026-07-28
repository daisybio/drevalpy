"""Preload and discovered-hyperparameter helpers for literature predictors."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import FeatureDataset

DISCOVERED_HYPERPARAMETERS_KEY = "discovered_hyperparameters"

CELL_LINE_PRELOAD_ATTRS: tuple[str, ...] = (
    "layer_connections",
    "gene2id_mapping_ont",
    "ontology_gene_order",
    "gene_dim_input",
    "model",
)


def merge_preload_hyperparameters(
    hyperparameters: dict[str, Any],
    preload_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply preload blobs and discovered hyperparameters before training."""
    merged = dict(hyperparameters)
    preload = dict(preload_state)
    discovered = preload.pop(DISCOVERED_HYPERPARAMETERS_KEY, None)
    if isinstance(discovered, dict):
        merged.update(discovered)
    return merged, preload


def collect_cell_line_preload(
    algorithm: LiteratureTrainingMixin,
    *,
    attrs: tuple[str, ...] = CELL_LINE_PRELOAD_ATTRS,
) -> dict[str, Any]:
    """Snapshot cell-line preload attributes from a trained algorithm."""
    return {name: getattr(algorithm, name) for name in attrs if getattr(algorithm, name, None) is not None}


def collect_drug_preload(
    algorithm: LiteratureTrainingMixin,
    seed_hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Return hyperparameters discovered while loading drug features."""
    discovered = {
        key: value
        for key, value in dict(algorithm.hyperparameters).items()
        if key not in seed_hyperparameters or seed_hyperparameters[key] != value
    }
    if not discovered:
        return {}
    return {DISCOVERED_HYPERPARAMETERS_KEY: discovered}


def load_dataset_cell_line_features(
    algorithm_cls: type[LiteratureTrainingMixin],
    data_path: str,
    dataset_name: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
    preload_attrs: tuple[str, ...] = CELL_LINE_PRELOAD_ATTRS,
) -> tuple[FeatureDataset, dict[str, Any]]:
    """Load cell-line features and preload state for a dataset."""
    algorithm = algorithm_cls()
    if hyperparameters:
        algorithm.hyperparameters = dict(hyperparameters)
    features = algorithm.load_cell_line_features(data_path, dataset_name)
    return features, collect_cell_line_preload(algorithm, attrs=preload_attrs)


def load_dataset_drug_features(
    algorithm_cls: type[LiteratureTrainingMixin],
    data_path: str,
    dataset_name: str,
    *,
    hyperparameters: dict[str, Any] | None = None,
) -> tuple[FeatureDataset | None, dict[str, Any]]:
    """Load drug features and any discovered hyperparameters for a dataset."""
    seed = dict(hyperparameters) if hyperparameters else {}
    algorithm = algorithm_cls()
    if seed:
        algorithm.hyperparameters = dict(seed)
    features = algorithm.load_drug_features(data_path, dataset_name)
    return features, collect_drug_preload(algorithm, seed)

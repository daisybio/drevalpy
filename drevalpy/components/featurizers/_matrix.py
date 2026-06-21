"""Helpers for building dense matrices from :class:`~drevalpy.datasets.dataset.FeatureDataset`."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import FeatureDataset


def unique_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Return unique entity ids in first-seen order."""
    uniq, index = np.unique(entity_ids, return_index=True)
    return uniq[index.argsort()]


def entity_index_map(entity_ids: np.ndarray) -> dict[str, int]:
    """Map entity id strings to row indices in a dense featurization matrix."""
    return {str(entity_id): row for row, entity_id in enumerate(entity_ids)}


def _entity_views(features: FeatureDataset, entity_id) -> dict:
    entity_key = str(entity_id)
    if entity_key in features.features:
        return features.features[entity_key]
    if entity_id in features.features:
        return features.features[entity_id]
    msg = f"Entity {entity_key!r} not found in FeatureDataset"
    raise KeyError(msg)


def stack_view_matrix(
    features: FeatureDataset,
    view: str,
    entity_ids: np.ndarray,
) -> np.ndarray:
    """Stack one view into ``(len(entity_ids), n_features)``."""
    rows: list[np.ndarray] = []
    for entity_id in entity_ids:
        entity_views = _entity_views(features, entity_id)
        if view not in entity_views:
            msg = f"View {view!r} not found for entity {str(entity_id)!r}"
            raise KeyError(msg)
        rows.append(np.asarray(entity_views[view], dtype=np.float64).ravel())
    return np.vstack(rows)


def stack_pair_features(
    cell_line_matrix: np.ndarray,
    drug_matrix: np.ndarray,
    cell_line_indices: np.ndarray,
    drug_indices: np.ndarray,
) -> np.ndarray:
    """Concatenate featurized cell-line and drug rows for each pair."""
    if cell_line_matrix.size == 0:
        return drug_matrix[drug_indices]
    if drug_matrix.size == 0:
        return cell_line_matrix[cell_line_indices]
    x_cell_line = cell_line_matrix[cell_line_indices]
    x_drug = drug_matrix[drug_indices]
    return np.concatenate([x_cell_line, x_drug], axis=1)

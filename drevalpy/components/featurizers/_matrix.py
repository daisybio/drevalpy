"""Helpers for building dense matrices from a `FeatureSource`."""

from __future__ import annotations

import numpy as np

from drevalpy.components.core.features.feature_source import FeatureSource


def unique_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Return unique entity ids in first-seen order.

    :param entity_ids: entity ids.
    :returns: Result.
    """
    uniq, index = np.unique(entity_ids, return_index=True)
    return uniq[index.argsort()]


def entity_index_map(entity_ids: np.ndarray) -> dict[str, int]:
    """Map entity id strings to row indices in a dense featurization matrix.

    :param entity_ids: entity ids.
    :returns: Result.
    """
    return {str(entity_id): row for row, entity_id in enumerate(entity_ids)}


def feature_names_for_view(source: FeatureSource, view: str) -> tuple[str, ...] | None:
    """Return ordered feature names for *view* via the source protocol.

    :param source: Feature source implementing the FeatureSource protocol.
    :param view: view.
    :returns: Result.
    """
    return source.get_feature_names(view)


def stack_view_matrix(
    source: FeatureSource,
    view: str,
    entity_ids: np.ndarray,
) -> np.ndarray:
    """Stack one view into ``(len(entity_ids), n_features)`` via the source protocol.

    :param source: Feature source implementing the FeatureSource protocol.
    :param view: view.
    :param entity_ids: entity ids.
    :returns: Result.
    """
    return source.get_view_matrix(view, entity_ids)


def stack_pair_features(
    cell_line_matrix: np.ndarray,
    drug_matrix: np.ndarray,
    cell_line_indices: np.ndarray,
    drug_indices: np.ndarray,
) -> np.ndarray:
    """Concatenate featurized cell-line and drug rows for each pair.

    :param cell_line_matrix: cell line matrix.
    :param drug_matrix: drug matrix.
    :param cell_line_indices: cell line indices.
    :param drug_indices: drug indices.
    :returns: Result.
    """
    if cell_line_matrix.size == 0:
        return drug_matrix[drug_indices]
    if drug_matrix.size == 0:
        return cell_line_matrix[cell_line_indices]
    x_cell_line = cell_line_matrix[cell_line_indices]
    x_drug = drug_matrix[drug_indices]
    return np.concatenate([x_cell_line, x_drug], axis=1)

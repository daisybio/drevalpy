"""Helpers for building dense matrices from a `FeatureSource`."""

from __future__ import annotations

import numpy as np

from drevalpy.types.data.feature_source import FeatureSource


def unique_entity_ids(entity_ids: np.ndarray) -> np.ndarray:
    """Return unique entity ids in first-seen order."""
    uniq, index = np.unique(entity_ids, return_index=True)
    return uniq[index.argsort()]


def feature_names_for_view(source: FeatureSource, view: str) -> tuple[str, ...] | None:
    """Return ordered feature names for *view* via the source protocol.

    :param source: Feature source implementing the FeatureSource protocol.
    :param view: View name to look up.
    :returns: Feature names in column order, or ``None`` when the source has none for the view.
    """
    return source.get_feature_names(view)


def stack_view_matrix(
    source: FeatureSource,
    view: str,
    entity_ids: np.ndarray,
) -> np.ndarray:
    """Stack one view into ``(len(entity_ids), n_features)`` via the source protocol.

    :param source: Feature source implementing the FeatureSource protocol.
    :param view: View name to stack.
    :param entity_ids: Identifiers selecting and ordering the rows.
    :returns: Matrix with one row per identifier.
    """
    return source.get_view_matrix(view, entity_ids)

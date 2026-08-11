"""FeatureSource ABC and Dataset adapter classes.

``FeatureSource`` is the abstract base class for feature access consumed by
featurizers. It provides the shared Dataset-backed logic (init, identifiers,
mdata, get_metadata) so concrete adapters only implement entity-specific
dispatch methods.

``CellLineFeatureSource`` and ``DrugFeatureSource`` are the concrete adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from drevalpy.types.data.dataset import Dataset


class FeatureSource(ABC):
    """Abstract base class for feature access consumed by featurizers.

    Provides shared Dataset-backed initialisation and common accessors.
    Subclasses only need to implement ``get_view_matrix``,
    ``get_feature_names``, and ``get_entity_view``.
    """

    def __init__(self, dataset: Dataset, entity_ids: np.ndarray) -> None:
        """Wrap a Dataset for entity feature access.

        Args:
            dataset: The backing dataset.
            entity_ids: Entity IDs this source exposes.
        """
        self._dataset = dataset
        self._ids = np.asarray(entity_ids, dtype=str)

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return self._ids

    @property
    def mdata(self) -> Any:
        """Underlying MuData object."""
        return self._dataset.mdata

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata from the underlying Dataset."""
        return self._dataset.get_uns(key)

    @abstractmethod
    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return (len(ids), n_features) float array for a dense numeric view."""
        ...

    @abstractmethod
    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return ordered feature/column names for a view, or None."""
        ...

    @abstractmethod
    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return the raw per-entity object for non-numeric views (graphs, etc.)."""
        ...


class CellLineFeatureSource(FeatureSource):
    """Adapts Dataset for cell-line featurizers."""

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return feature matrix for the given cell lines and view."""
        return self._dataset.get_cell_line_features(view, entity_ids)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return feature names for a cell-line view."""
        return self._dataset.get_cell_line_feature_names(view)

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return a per-entity value for a single cell line.

        For metadata keys like "tissue", delegates to Dataset.get_tissue().
        For omics modalities, returns the feature vector from that modality.
        """
        if view == "tissue":
            return self._dataset.get_tissue(np.array([entity_id]))[0]
        return self._dataset.get_cell_line_features(view, np.array([entity_id]))[0]


class DrugFeatureSource(FeatureSource):
    """Adapts Dataset for drug featurizers."""

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return feature matrix for the given drugs and view."""
        return self._dataset.get_drug_features(view, entity_ids)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return feature names for a drug view."""
        return self._dataset.get_drug_feature_names(view)

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return a per-entity value for a single drug.

        For graph views (stored in mdata.uns["drug_graphs"]), returns the graph
        dict. For other views (varm-backed embeddings), returns the feature vector.
        """
        if view == "drug_graph":
            graphs = self._dataset.get_drug_graphs(np.array([entity_id]))
            return graphs[0] if graphs else None
        return self._dataset.get_drug_features(view, np.array([entity_id]))[0]

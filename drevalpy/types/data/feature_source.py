"""FeatureSource protocol and Dataset adapter classes.

``FeatureSource`` defines the minimal feature-access interface consumed by
featurizers. ``CellLineFeatureSource`` and ``DrugFeatureSource`` are thin
adapters that wrap a ``Dataset`` and implement the protocol for the
respective entity type.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from drevalpy.types.data.dataset import Dataset


@runtime_checkable
class FeatureSource(Protocol):
    """Minimal feature-access interface consumed by featurizers.

    Provides the typed interface accepted by Featurizer.fit/transform.
    Both Dataset (via a thin wrapper) and test mocks can satisfy this.
    """

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        ...

    @property
    def mdata(self) -> Any:
        """Underlying MuData object, or None if unavailable."""
        ...

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return (len(ids), n_features) float array for a dense numeric view."""
        ...

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return ordered feature/column names for a view, or None."""
        ...

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return the raw per-entity object for non-numeric views (graphs, etc.)."""
        ...

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata (e.g. ontology structures)."""
        ...


class CellLineFeatureSource:
    """Adapts Dataset for cell-line featurizers."""

    def __init__(self, dataset: Dataset, cell_line_ids: np.ndarray) -> None:
        """Wrap a Dataset for cell-line feature access.

        Args:
            dataset: The backing dataset.
            cell_line_ids: Cell-line IDs this source exposes.
        """
        self._dataset = dataset
        self._ids = np.asarray(cell_line_ids, dtype=str)

    @property
    def identifiers(self) -> np.ndarray:
        """All available cell-line IDs."""
        return self._ids

    @property
    def mdata(self) -> Any:
        """Underlying MuData object."""
        return self._dataset.mdata

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

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata from the underlying Dataset."""
        return self._dataset.get_uns(key)


class DrugFeatureSource:
    """Adapts Dataset for drug featurizers."""

    def __init__(self, mudataset: Dataset, drug_ids: np.ndarray) -> None:
        """Wrap a Dataset for drug feature access.

        Args:
            mudataset: The backing dataset.
            drug_ids: Drug IDs this source exposes.
        """
        self._mu = mudataset
        self._ids = np.asarray(drug_ids, dtype=str)

    @property
    def identifiers(self) -> np.ndarray:
        """All available drug IDs."""
        return self._ids

    @property
    def mdata(self) -> Any:
        """Underlying MuData object."""
        return self._mu.mdata

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return feature matrix for the given drugs and view."""
        return self._mu.get_drug_features(view, entity_ids)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return feature names for a drug view."""
        return self._mu.get_drug_feature_names(view)

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return a per-entity value for a single drug.

        For graph views (stored in mdata.uns["drug_graphs"]), returns the graph
        dict. For other views (varm-backed embeddings), returns the feature vector.
        """
        if view == "drug_graph":
            graphs = self._mu.get_drug_graphs(np.array([entity_id]))
            return graphs[0] if graphs else None
        return self._mu.get_drug_features(view, np.array([entity_id]))[0]

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata from the underlying Dataset."""
        return self._mu.get_uns(key)

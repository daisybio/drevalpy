"""PCA cell-line featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._matrix import feature_names_for_view
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "pca",
    description="PCA compression of one dense cell-line view fit on training cell lines.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PCACellLineFeaturizer(DenseViewCellLineFeaturizer):
    """Reduce one cell-line view with PCA."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str, n_components: int = 128) -> None:
        """Initialize instance state.

        :param view: view.
        :param n_components: n components.
        :raises ValueError: Raised on invalid input.
        """
        from sklearn.decomposition import PCA

        if not view or not view.strip():
            msg = "pca featurizer requires an explicit view"
            raise ValueError(msg)
        super().__init__(view=view)
        self._n_components = int(n_components)
        self._pca = PCA(n_components=self._n_components)
        self._feature_names: tuple[str, ...] | None = None

    def _fetch_hyperparameters(self) -> dict[str, Any]:
        """Match only a variant stored with this component count.

        :returns: HP mapping identifying the stored variant.
        """
        return {"n_components": self._n_components}

    def _on_precomputed_fit(self, source: FeatureSource) -> None:
        """Record the source's feature names alongside a stored variant.

        :param source: Feature source carrying the stored variant.
        """
        self._feature_names = feature_names_for_view(source, self._view)

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit PCA on the training rows and return the retained component count.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Cell-line identifiers to fit on.
        :returns: Number of retained components.
        """
        matrix = self._raw_matrix(source, entity_ids)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pca.n_components = n_components
        self._pca.fit(matrix)
        self._feature_names = feature_names_for_view(source, self._view)
        return n_components

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Project *matrix* through the fitted PCA, realigning columns by name first.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: PCA-reduced feature matrix.
        """
        names = feature_names_for_view(source, self._view)
        if self._feature_names is not None and names is not None:
            source_indices = {name: index for index, name in enumerate(names)}
            aligned = np.zeros((matrix.shape[0], len(self._feature_names)), dtype=matrix.dtype)
            for index, name in enumerate(self._feature_names):
                source_index = source_indices.get(name)
                if source_index is not None:
                    aligned[:, index] = matrix[:, source_index]
            matrix = aligned
        return self._pca.transform(matrix)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "n_components": {"type": "int", "low": 8, "high": 512, "default": 128},
        }

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {
            "pca": self._pca,
            "view": self._view,
            "n_components": self._n_components,
            "output_dim": self._output_dim,
            "feature_names": self._feature_names,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        from sklearn.decomposition import PCA

        pca = state.get("pca")
        if isinstance(pca, PCA):
            self._pca = pca
        n_components = state.get("n_components")
        if isinstance(n_components, int):
            self._n_components = n_components
        feature_names = state.get("feature_names")
        if isinstance(feature_names, tuple):
            self._feature_names = tuple(str(name) for name in feature_names)
        self._restore_dense_state(state)

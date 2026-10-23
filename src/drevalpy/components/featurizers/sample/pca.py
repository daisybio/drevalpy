"""PCA sample featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "pca",
    description="PCA compression of one dense sample view fit on training samples.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PCASampleFeaturizer(DenseViewSampleFeaturizer):
    """Reduce one sample view with PCA."""

    requires_view: ClassVar[bool] = True

    def __init__(self, *, view: str, n_components: int = 128) -> None:
        """Bind the view to reduce and the target dimension.

        :param view: Name of the sample view to reduce.
        :param n_components: Requested component count; capped at fit time by the number of
            fit samples and view features.
        :raises ValueError: If *view* is empty or blank.
        """
        from sklearn.decomposition import PCA

        if not view or not view.strip():
            msg = "pca featurizer requires an explicit view"
            raise ValueError(msg)
        super().__init__(view=view)
        self._n_components = int(n_components)
        self._pca = PCA(n_components=self._n_components)

    def _fetch_hyperparameters(self) -> dict[str, Any]:
        """Match only a variant stored with this component count.

        :returns: HP mapping identifying the stored variant.
        """
        return {"n_components": self._n_components}

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit PCA on the training rows and return the retained component count.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Sample identifiers to fit on.
        :returns: Number of retained components.
        """
        matrix = self._raw_matrix(source, entity_ids)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pca.n_components = n_components
        self._pca.fit(matrix)
        return n_components

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Project *matrix*, already in fitted column order, through the fitted PCA.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: PCA-reduced feature matrix.
        """
        _ = source
        return self._pca.transform(matrix)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the searchable hyperparameters."""
        return {
            "n_components": {"type": "int", "low": 8, "high": 512, "default": 128},
        }

    def get_state(self) -> dict[str, object]:
        """Return the fitted PCA, the requested component count and the dense state."""
        return {
            "pca": self._pca,
            "n_components": self._n_components,
            **self._dense_state(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        from sklearn.decomposition import PCA

        pca = state.get("pca")
        if isinstance(pca, PCA):
            self._pca = pca
        n_components = state.get("n_components")
        if isinstance(n_components, int):
            self._n_components = n_components
        self._restore_dense_state(state)

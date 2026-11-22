"""PCA sample featurizer.

The view is standardized (centred, scaled to unit variance per feature) and then reduced, both fitted on the
training samples only. Standardization is part of the featurizer, not a step a recipe has to put in front of
it: PCA ranks directions by variance, so on unscaled data the features with the largest unit dominate every
component. A feature that is constant in the training samples is centred and left unscaled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.registry.featurizer import register
from drevalpy.types.data.feature_block import FeaturizerOutput
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@register("pca", side=Side.SAMPLE, description="PCA compression of one dense sample view fit on training samples.")
class PCASampleFeaturizer(Featurizer):
    """Reduce one sample view with PCA after standardizing it."""

    def __init__(self, *, view: str, n_components: int = 128) -> None:
        """Bind the view to reduce and the target dimension.

        :param view: Name of the sample view to reduce.
        :param n_components: Requested component count; capped at fit time by the number of
            fit samples and view features.
        :raises ValueError: If *view* is empty or blank.
        """
        if not view or not view.strip():
            msg = "pca featurizer requires an explicit view"
            raise ValueError(msg)
        self._view = view
        self._n_components = int(n_components)
        self._names: tuple[str, ...] | None = None
        self._pipeline: Pipeline | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the searchable hyperparameters."""
        return {
            "n_components": {"type": "int", "low": 8, "high": 512, "default": 128},
        }

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit the standardization and PCA on the training rows, capping the component count by the matrix shape.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        _ = train_pair_ids, es_pair_ids
        self._names = source.feature_names(self._view)
        matrix = view_matrix(source, self._view, train_ids, self._names)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit(matrix)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        """Standardize and project the view's rows for *ids* through the fitted PCA.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One block of components, named after the view.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._pipeline is None:
            msg = "PCASampleFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            matrix = np.empty((0, int(self._pipeline[-1].n_components_)))
        else:
            matrix = self._pipeline.transform(view_matrix(source, self._view, ids, self._names))
        return dense_result(ids, self._view, matrix, None)

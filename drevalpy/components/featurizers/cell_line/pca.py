"""PCA cell-line featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource


@register(
    "pca",
    description="PCA compression of one dense cell-line view fit on training cell lines.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class PCACellLineFeaturizer(CellLineFeaturizer):
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
        self._view = view
        self._n_components = int(n_components)
        self._pca = PCA(n_components=self._n_components)
        self._output_dim = 0
        self._feature_names: tuple[str, ...] | None = None

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> PCACellLineFeaturizer:
        """Fit on training data.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Result.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        mdata = getattr(source, "mdata", None)
        precomputed = (
            self.fetch(mdata, ids, hyperparameters={"n_components": self._n_components}) if mdata is not None else None
        )
        if precomputed is not None:
            self._output_dim = int(precomputed.shape[1])
            self._feature_names = feature_names_for_view(source, self._view)
            return self
        matrix = stack_view_matrix(source, self._view, ids)
        n_components = min(self._n_components, matrix.shape[0], matrix.shape[1])
        self._pca.n_components = n_components
        self._pca.fit(matrix)
        self._output_dim = n_components
        self._feature_names = feature_names_for_view(source, self._view)
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs through fitted PCA.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: PCA-reduced feature matrix.
        """
        mdata = getattr(source, "mdata", None)
        precomputed = (
            self.fetch(mdata, entity_ids, hyperparameters={"n_components": self._n_components})
            if mdata is not None
            else None
        )
        if precomputed is not None:
            return precomputed.astype(np.float32)
        matrix = stack_view_matrix(source, self._view, entity_ids)
        names = feature_names_for_view(source, self._view)
        if self._feature_names is not None and names is not None:
            source_indices = {name: index for index, name in enumerate(names)}
            aligned = np.zeros((len(entity_ids), len(self._feature_names)), dtype=matrix.dtype)
            for index, name in enumerate(self._feature_names):
                source_index = source_indices.get(name)
                if source_index is not None:
                    aligned[:, index] = matrix[:, source_index]
            matrix = aligned
        return self._pca.transform(matrix).astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._view: numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=feature_names_for_view(source, self._view),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

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
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        n_components = state.get("n_components")
        if isinstance(n_components, int):
            self._n_components = n_components
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        feature_names = state.get("feature_names")
        if isinstance(feature_names, tuple):
            self._feature_names = tuple(str(name) for name in feature_names)

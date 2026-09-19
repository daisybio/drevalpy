"""SuperFELTR multi-omics feature-selection featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register(
    "superfeltrOmics",
    description="SuperFELTR variance-selected multi-omics inputs.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SuperFELTROmicsFeaturizer(CellLineFeaturizer):
    """Select high-variance features independently in each SuperFELTR view."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(self, *, n_features_per_view: int = 1000) -> None:
        """Store per-view variance-selection feature count and initialize selectors.

        :param n_features_per_view: Number of features to keep per omics view.
        """
        self._n_features = int(n_features_per_view)
        self._masks: dict[str, np.ndarray] = {}
        self._feature_names: dict[str, tuple[str, ...]] = {}

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> SuperFELTROmicsFeaturizer:
        """Fit variance selectors independently in each omics view.

        :param source: Feature source providing view matrices.
        :param entity_ids: Optional explicit fit ids.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = np.unique(entity_ids if entity_ids is not None else source.identifiers)
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, ids) if mdata is not None else None
        if precomputed is not None:
            names = source.get_feature_names("gene_expression")
            self._masks = {"gene_expression": np.ones(precomputed.shape[1], dtype=bool)}
            self._feature_names = {"gene_expression": tuple(names) if names else ()}
            return self
        for view in _VIEWS:
            matrix = self._view_matrix(source, view, ids)
            variances = np.var(matrix, axis=0)
            mask = np.zeros(len(variances), dtype=bool)
            mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True
            self._masks[view] = mask

            names = source.get_feature_names(view)
            if names is not None:
                self._feature_names[view] = tuple(np.array(names)[mask])
            else:
                self._feature_names[view] = ()
        return self

    @staticmethod
    def _view_matrix(source: FeatureSource, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Apply the regression adaptation's expression transform before selection."""
        matrix = source.get_view_matrix(view, entity_ids)
        return np.arcsinh(matrix) if view == "gene_expression" else matrix

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Return per-omics numeric blocks with variance-selected columns.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            blocks = {
                "gene_expression": numeric_feature_block(
                    precomputed.astype(np.float32),
                    feature_names=self._feature_names.get("gene_expression"),
                )
            }
            return FeaturizerResult(blocks["gene_expression"].values, blocks)
        blocks = {
            view: numeric_feature_block(
                self._view_matrix(source, view, entity_ids)[:, mask].astype(np.float32),
                feature_names=self._feature_names.get(view),
            )
            for view, mask in self._masks.items()
        }
        return FeaturizerResult(blocks["gene_expression"].values, blocks)

    def _empty_transform_features(self, source: FeatureSource) -> FeaturizerResult:
        """Build selected omics schemas without invoking a transformer.

        :param source: Unused; fitted masks supply dimensions.
        :returns: Empty gene expression matrix and all omics blocks.
        """
        blocks = {
            view: numeric_feature_block(
                np.empty((0, int(mask.sum())), dtype=np.float32), feature_names=self._feature_names.get(view)
            )
            for view, mask in self._masks.items()
        }
        return FeaturizerResult(blocks["gene_expression"].values, blocks)

    @property
    def output_dim(self) -> int:
        """Return total selected features across all views.

        :returns: Sum of selected features in every view.
        """
        return int(sum(mask.sum() for mask in self._masks.values()))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable per-view variance-selection feature count.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"n_features_per_view": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        """Serialize masks and feature-name metadata.

        :returns: Fitted state mapping.
        """
        return {
            "masks": self._masks,
            "feature_names": self._feature_names,
            "n_features_per_view": self._n_features,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore masks and feature names from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        masks = state.get("masks")
        if isinstance(masks, dict) and all(isinstance(value, np.ndarray) for value in masks.values()):
            self._masks = {str(key): value for key, value in masks.items()}
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): tuple(value) for key, value in names.items()}

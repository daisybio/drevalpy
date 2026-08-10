"""SuperFELTR multi-omics feature-selection featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register_cell_line_featurizer(
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
            for view in _VIEWS:
                names = source.get_feature_names(view)
                self._masks[view] = np.ones(1, dtype=bool)
                self._feature_names[view] = tuple(names) if names else ()
            return self
        for view in _VIEWS:
            matrix = source.get_view_matrix(view, ids)
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

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return variance-selected gene-expression features only.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of selected gene-expression features.
        """
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        mask = self._masks["gene_expression"]
        return source.get_view_matrix("gene_expression", entity_ids)[:, mask].astype(np.float32)

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return per-omics numeric blocks with variance-selected columns.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return {
                "gene_expression": numeric_feature_block(
                    precomputed.astype(np.float32),
                    feature_names=self._feature_names.get("gene_expression"),
                )
            }
        return {
            view: numeric_feature_block(
                source.get_view_matrix(view, entity_ids)[:, mask].astype(np.float32),
                feature_names=self._feature_names.get(view),
            )
            for view, mask in self._masks.items()
        }

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

"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.preprocessing import StandardScaler

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register_cell_line_featurizer(
    "molirOmics",
    description="MOLIR multi-omics input preparation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class MOLIROmicsFeaturizer(CellLineFeaturizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    learned: ClassVar[bool] = True
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS

    def __init__(self, *, n_gene_expression_features: int = 1000) -> None:
        """Store the variance-selection feature count and initialize scalers.

        :param n_gene_expression_features: Number of gene-expression features to keep.
        """
        self._n_features = int(n_gene_expression_features)
        self._scaler = StandardScaler()
        self._mask: np.ndarray = np.array([], dtype=bool)
        self._selected_feature_names: tuple[str, ...] = ()
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> MOLIROmicsFeaturizer:
        """Arcsinh-scale gene expression and fit variance selection on training ids.

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
            self._mask = np.ones(precomputed.shape[1], dtype=bool)
            self._feature_names = {"gene_expression": source.get_feature_names("gene_expression")}
            for view in _VIEWS[1:]:
                self._feature_names[view] = source.get_feature_names(view)
            return self
        matrix = np.arcsinh(source.get_view_matrix("gene_expression", ids))
        self._scaler.fit(matrix)
        scaled = self._scaler.transform(matrix)
        variances = np.var(scaled, axis=0)
        self._mask = np.zeros(len(variances), dtype=bool)
        self._mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True

        ge_names = source.get_feature_names("gene_expression")
        if ge_names is not None:
            self._selected_feature_names = tuple(np.array(ge_names)[self._mask])
        else:
            self._selected_feature_names = ()

        self._feature_names = {
            "gene_expression": self._selected_feature_names,
            **{view: source.get_feature_names(view) for view in _VIEWS[1:]},
        }
        return self

    def _gene_expression(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        mdata = getattr(source, "mdata", None)
        precomputed = self.fetch(mdata, entity_ids) if mdata is not None else None
        if precomputed is not None:
            return precomputed.astype(np.float32)
        matrix = np.arcsinh(source.get_view_matrix("gene_expression", entity_ids))
        return self._scaler.transform(matrix)[:, self._mask].astype(np.float32)

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return scaled, variance-selected gene-expression features.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Float matrix of selected gene-expression features.
        """
        return self._gene_expression(source, entity_ids)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return per-omics numeric blocks for MOLIR.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        return {
            "gene_expression": numeric_feature_block(
                self._gene_expression(source, entity_ids), feature_names=self._feature_names.get("gene_expression")
            ),
            **{
                view: numeric_feature_block(
                    source.get_view_matrix(view, entity_ids).astype(np.float32),
                    feature_names=self._feature_names.get(view),
                )
                for view in _VIEWS[1:]
            },
        }

    @property
    def output_dim(self) -> int:
        """Return the number of selected gene-expression features.

        :returns: Selected gene-expression feature count.
        """
        return int(self._mask.sum()) if self._mask.size > 0 else 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable variance-selection feature count.

        :returns: Ray Tune-style hyperparameter space mapping.
        """
        return {"n_gene_expression_features": {"type": "int", "low": 1, "high": 1000, "default": 1000}}

    def get_state(self) -> dict[str, object]:
        """Serialize scaler, mask, and feature-name metadata.

        :returns: Fitted state mapping.
        """
        return {
            "scaler": self._scaler,
            "mask": self._mask,
            "selected_feature_names": self._selected_feature_names,
            "feature_names": self._feature_names,
            "n_gene_expression_features": self._n_features,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore scaler, mask, and feature names from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        scaler = state.get("scaler")
        if isinstance(scaler, StandardScaler):
            self._scaler = scaler
        mask = state.get("mask")
        if isinstance(mask, np.ndarray):
            self._mask = mask
        selected = state.get("selected_feature_names")
        if isinstance(selected, tuple):
            self._selected_feature_names = selected
        names = state.get("feature_names")
        if isinstance(names, dict):
            self._feature_names = {str(key): value for key, value in names.items()}

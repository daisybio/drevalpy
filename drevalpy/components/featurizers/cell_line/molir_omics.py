"""MOLIR multi-omics preprocessing featurizer."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import DenseViewCellLineFeaturizer
from drevalpy.registry.cell_line_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


@register(
    "molirOmics",
    description="MOLIR multi-omics input preparation.",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class MOLIROmicsFeaturizer(DenseViewCellLineFeaturizer):
    """Arcsinh-scale and select variable gene-expression features for MOLIR."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("mutations", FeatureFormat.NUMERIC_MATRIX),
        BlockSpec("copy_number_variation_gistic", FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = _VIEWS
    fit_on_unique_ids: ClassVar[bool] = True

    def __init__(self, *, n_gene_expression_features: int = 1000) -> None:
        """Store the variance-selection feature count and initialize scalers.

        :param n_gene_expression_features: Number of gene-expression features to keep.
        """
        from sklearn.preprocessing import StandardScaler

        super().__init__()
        self._n_features = int(n_gene_expression_features)
        self._scaler = StandardScaler()
        self._mask: np.ndarray = np.array([], dtype=bool)
        self._selected_feature_names: tuple[str, ...] = ()
        self._feature_names: dict[str, tuple[str, ...] | None] = {}

    def _on_precomputed_fit(self, source: FeatureSource) -> None:
        """Accept every stored column and record the per-view feature names.

        :param source: Feature source carrying the stored variant.
        """
        self._mask = np.ones(self._output_dim, dtype=bool)
        self._feature_names = {view: source.get_feature_names(view) for view in _VIEWS}

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Fit the scaler and pick the highest-variance gene-expression features.

        :param source: Feature source providing view matrices.
        :param entity_ids: Deduplicated cell-line identifiers to fit on.
        :returns: Number of selected gene-expression features.
        """
        matrix = np.arcsinh(self._raw_matrix(source, entity_ids))
        self._scaler.fit(matrix)
        variances = np.var(self._scaler.transform(matrix), axis=0)
        self._mask = np.zeros(len(variances), dtype=bool)
        self._mask[np.argsort(variances)[::-1][: min(self._n_features, len(variances))]] = True

        ge_names = source.get_feature_names("gene_expression")
        self._selected_feature_names = () if ge_names is None else tuple(np.array(ge_names)[self._mask])
        self._feature_names = {
            "gene_expression": self._selected_feature_names,
            **{view: source.get_feature_names(view) for view in _VIEWS[1:]},
        }
        return int(self._mask.sum())

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Arcsinh-scale *matrix* and keep only the selected features.

        :param source: Feature source the matrix came from.
        :param matrix: Raw gene-expression matrix.
        :returns: Selected gene-expression features.
        """
        _ = source
        return self._scaler.transform(np.arcsinh(matrix))[:, self._mask]

    def _block_feature_names(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Return the selected gene names recorded at fit time.

        :param source: Feature source (unused; names are recorded during fit).
        :returns: Selected gene-expression feature names, or ``None``.
        """
        _ = source
        return self._feature_names.get("gene_expression")

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return per-omics numeric blocks for MOLIR.

        The gene-expression block goes through the shared dense path; the two other
        omics views are passed through untouched.

        :param source: Feature source providing view matrices.
        :param entity_ids: Cell-line identifiers to transform.
        :returns: Mapping of omics view name to numeric blocks.
        """
        return {
            **super()._transform_blocks(source, entity_ids),
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
        from sklearn.preprocessing import StandardScaler

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
